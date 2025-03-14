import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph, GCNConv
from torch_scatter import scatter_max, scatter_mean
from torch_sparse import SparseTensor
import math


def init_weight_(m):
    """Инициализация весов Xavier/Glorot для сверточных и линейных слоев."""
    if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class PositionalEncoding(nn.Module):
    """Позиционное кодирование для трансформера."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        pos_encoding = self.pe[:, :x.size(1)].repeat(x.size(0), 1, 1)
        x = x + pos_encoding
        return self.dropout(x)


class SelfAttention(nn.Module):
    """Модуль самовнимания с возможностью учета локальной структуры."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_local=True, k=16):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.use_local = use_local
        self.k = k

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.apply(init_weight_)

    def forward(self, x, xyz=None, batch=None):
        """
        Args:
            x: Tensor, shape [batch_size*seq_len, embedding_dim]
            xyz: Опционально, координаты точек для локального внимания [batch_size*seq_len, 3]
            batch: Опционально, индексы батча для точек [batch_size*seq_len]
        """
        B_seq_len, C = x.shape
        
        # QKV проекции
        qkv = self.qkv(x).reshape(B_seq_len, 3, self.num_heads, C // self.num_heads).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B_seq_len, num_heads, head_dim]
        
        # Если используем локальную структуру и даны координаты
        if self.use_local and xyz is not None and batch is not None:
            # Получаем k ближайших соседей для каждой точки
            edge_index = knn_graph(xyz, k=self.k, batch=batch)
            row, col = edge_index
            
            # Для каждой точки и ее соседей вычисляем внимание
            q_i = q[row]  # [edges, num_heads, head_dim]
            k_j = k[col]  # [edges, num_heads, head_dim]
            
            # Вычисляем веса внимания между точками и их соседями
            attn = (q_i * k_j).sum(dim=-1) * self.scale  # [edges, num_heads]
            attn = F.softmax(attn, dim=0)  # нормализуем по соседям
            attn = self.attn_drop(attn)
            
            # Применяем веса к значениям
            v_j = v[col]  # [edges, num_heads, head_dim]
            weighted_v = v_j * attn.unsqueeze(-1)  # [edges, num_heads, head_dim]
            
            # Суммируем взвешенные значения для каждой точки
            out = torch.zeros_like(v)  # [B_seq_len, num_heads, head_dim]
            for i in range(B_seq_len):
                mask = row == i
                if mask.any():
                    out[i] = weighted_v[mask].sum(dim=0)
            
        else:
            # Стандартное внимание по всем точкам
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B_seq_len, num_heads, B_seq_len]
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v  # [B_seq_len, num_heads, head_dim]
        
        # Объединяем результаты со всех головок внимания
        out = out.reshape(B_seq_len, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out


class FeedForward(nn.Module):
    """Слой прямого распространения трансформера с GeLU активацией."""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.apply(init_weight_)
        
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Блок трансформера с самовниманием и Feed-Forward слоем."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 use_local=True, k=16, pre_norm=True):
        super().__init__()
        self.pre_norm = pre_norm
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop, use_local=use_local, k=k
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), dropout=drop)
        
    def forward(self, x, xyz=None, batch=None):
        if self.pre_norm:
            # Pre-LayerNorm архитектура (как в оригинальном Transformer)
            identity = x
            x = self.norm1(x)
            x = self.attn(x, xyz, batch) + identity
            
            identity = x
            x = self.norm2(x)
            x = self.mlp(x) + identity
        else:
            # Post-LayerNorm архитектура (как в BERT)
            x = self.norm1(x + self.attn(x, xyz, batch))
            x = self.norm2(x + self.mlp(x))
        
        return x


class FeatureTransformer(nn.Module):
    """Извлекает признаки из облака точек с использованием трансформера."""
    def __init__(self, k=8, dim=128, depth=3, num_heads=8, mlp_ratio=4.0, drop_rate=0.1, use_local=True):
        super().__init__()
        self.k = k
        self.dim = dim
        
        # Начальное преобразование признаков
        self.input_embed = nn.Sequential(
            nn.Conv1d(3, dim//2, 1),
            nn.BatchNorm1d(dim//2),
            nn.ReLU(),
            nn.Conv1d(dim//2, dim, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU()
        )
        
        # Слой позиционного кодирования
        self.pos_encoding = PositionalEncoding(dim, dropout=drop_rate)
        
        # Слои трансформера
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=drop_rate, use_local=use_local, k=k
            ) for _ in range(depth)
        ])
        
        self.apply(init_weight_)
        
    def forward(self, x):
        """
        Вход:  (B, 3, N) - координаты точек
        Выход: (B, C, N) - признаки точек
        """
        B, _, N = x.shape
        
        # Извлекаем начальные признаки через сверточную сеть
        features = self.input_embed(x)  # (B, dim, N)
        
        # Переформатируем для трансформера и добавляем позиционное кодирование
        features = features.transpose(1, 2)  # (B, N, dim)
        features = self.pos_encoding(features)
        features = features.reshape(B*N, self.dim)  # (B*N, dim)
        
        # Подготавливаем данные для модуля внимания
        x_flat = x.transpose(1, 2).reshape(B*N, 3)  # (B*N, 3)
        batch = torch.arange(B, device=x.device).repeat_interleave(N)  # (B*N,)
        
        # Обрабатываем через слои трансформера
        for block in self.transformer_blocks:
            features = block(features, x_flat, batch)
        
        # Возвращаем к исходному формату
        features = features.view(B, N, self.dim).transpose(1, 2)  # (B, dim, N)
        
        return features


class PointTransformerUnpool(nn.Module):
    """Модуль для апсемплинга облака точек с использованием трансформера."""
    def __init__(self, k=8, in_dim=128, dim=128, depth=2, num_heads=8, mlp_ratio=4.0, drop_rate=0.1):
        super().__init__()
        self.k = k
        
        # Блоки трансформера для обработки признаков
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=in_dim if i == 0 else dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio,
                drop=drop_rate, 
                attn_drop=drop_rate, 
                use_local=True, 
                k=k
            ) for i in range(depth)
        ])
        
        # Проекция размерности при необходимости
        self.dim_proj = nn.Identity() if in_dim == dim else nn.Linear(in_dim, dim)
        
        # Предсказание смещений для новых точек
        self.offset_predictor = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 6),  # 3 координаты * 2 новые точки
            nn.Tanh()  # Ограничиваем смещения в диапазоне [-1, 1]
        )
        
        self.apply(init_weight_)
        
    def forward(self, xyz, features):
        """
        Вход:  xyz - (B, 3, N), координаты точек
               features - (B, C, N), признаки точек
        Выход: new_xyz - (B, 3, 2*N), новые координаты
               new_features - (B, C, N), обновленные признаки
        """
        B, _, N = xyz.shape
        
        # Подготовка данных
        x_flat = xyz.transpose(1, 2).reshape(B*N, 3)  # (B*N, 3)
        h = features.transpose(1, 2).reshape(B*N, features.size(1))  # (B*N, C)
        batch = torch.arange(B, device=xyz.device).repeat_interleave(N)  # (B*N,)
        
        # Проекция размерности при необходимости
        h = self.dim_proj(h)
        
        # Проход через блоки трансформера
        for block in self.transformer_blocks:
            h = block(h, x_flat, batch)
        
        # Предсказываем смещения для новых точек
        offset = self.offset_predictor(h) * 0.1  # Масштабный коэффициент для стабильности
        
        # Создаем новые точки с предсказанными смещениями
        offset = offset.view(B, N, 3, 2).permute(0, 2, 3, 1)  # (B, 3, 2, N)
        xyz_expanded = xyz.unsqueeze(2).expand(-1, -1, 2, -1)  # (B, 3, 2, N)
        new_xyz = xyz_expanded + offset  # (B, 3, 2, N)
        new_xyz = new_xyz.reshape(B, 3, 2*N)  # (B, 3, 2*N)
        
        # Возвращаем обновленные признаки
        new_features = h.view(B, N, -1).transpose(1, 2)  # (B, C, N)
        
        return new_xyz, new_features


def knn_interpolate_robust(x, pos_x, pos_y, k, batch_x=None, batch_y=None):
    """
    Универсальная функция интерполяции признаков, совместимая с разными версиями PyTorch Geometric.
    
    Args:
        x: исходные признаки (n, c)
        pos_x: координаты исходных точек (n, 3)
        pos_y: координаты целевых точек (m, 3)
        k: количество соседей
        batch_x: индексы батча для исходных точек
        batch_y: индексы батча для целевых точек
    """
    # Находим k ближайших соседей для каждой целевой точки
    assign_index = knn_graph(
        x=torch.cat([pos_y, pos_x]), 
        k=k, 
        batch=None if batch_y is None else torch.cat([batch_y, batch_x]),
        flow='target_to_source'
    )
    
    # Оставляем только связи между целевыми и исходными точками
    row, col = assign_index
    mask = row < pos_y.size(0)
    row, col = row[mask], col[mask]
    mask = col >= pos_y.size(0)
    row, col = row[mask], col[mask] - pos_y.size(0)
    
    # Вычисляем веса на основе расстояний
    dist = torch.norm(pos_y[row] - pos_x[col], p=2, dim=-1)
    
    # Инвертируем расстояния для получения весов
    dist = 1.0 / (dist + 1e-5)
    
    # Нормализуем веса для каждой целевой точки
    norm = scatter_mean(dist, row, dim=0, dim_size=pos_y.size(0))
    norm = norm[row]
    weight = dist / norm
    
    # Интерполируем признаки с помощью взвешенного суммирования
    y = scatter_mean(x[col] * weight.view(-1, 1), row, dim=0, dim_size=pos_y.size(0))
    
    return y


class FeatureNet(nn.Module):
    """Класс-обертка для обратной совместимости."""
    def __init__(self, k=8, dim=128, num_block=3):
        super().__init__()
        self.feature_transformer = FeatureTransformer(k=k, dim=dim, depth=num_block)
        
    def forward(self, x):
        return self.feature_transformer(x)


class TransformerGenerator(nn.Module):
    """Генератор облаков точек на основе архитектуры трансформера."""
    def __init__(self, cfg):
        super().__init__()
        self.k = cfg['k']
        self.feat_dim = cfg['feat_dim']
        self.res_conv_dim = cfg['res_conv_dim']
        
        # Сеть извлечения признаков на основе трансформера
        self.featurenet = FeatureTransformer(
            k=self.k, 
            dim=self.feat_dim, 
            depth=3,
            num_heads=8
        )
        
        # Две сети апсемплинга для 4-кратного увеличения плотности
        self.unpool_1 = PointTransformerUnpool(
            k=self.k, 
            in_dim=self.feat_dim, 
            dim=self.res_conv_dim,
            depth=2,
            num_heads=8
        )
        
        self.unpool_2 = PointTransformerUnpool(
            k=self.k, 
            in_dim=self.res_conv_dim, 
            dim=self.res_conv_dim,
            depth=2,
            num_heads=8
        )
        
    @torch.cuda.amp.autocast()  # Включаем смешанную точность для ускорения
    def forward(self, xyz):
        """
        Вход:  xyz - (B, 3, N), координаты точек
        Выход: upsampled_xyz - (B, 3, 4*N), координаты после апсемплинга
        """
        # 1. Извлекаем признаки
        features = self.featurenet(xyz)  # (B, C, N)
        
        # 2. Первый уровень апсемплинга (x2)
        xyz_x2, features_1 = self.unpool_1(xyz, features)  # (B, 3, 2*N), (B, C, N)
        
        # 3. Передаем признаки от исходных точек к новым с помощью интерполяции
        B, _, N_x2 = xyz_x2.shape
        N = xyz.shape[2]
        
        # Преобразуем в нужный формат для интерполяции
        xyz_flat = xyz.transpose(1, 2).reshape(-1, 3)  # (B*N, 3)
        xyz_x2_flat = xyz_x2.transpose(1, 2).reshape(-1, 3)  # (B*N_x2, 3)
        features_flat = features.transpose(1, 2).reshape(-1, features.size(1))  # (B*N, C)
        
        batch_old = torch.arange(B, device=xyz.device).repeat_interleave(N)  # (B*N,)
        batch_new = torch.arange(B, device=xyz.device).repeat_interleave(N_x2)  # (B*N_x2,)
        
        # Интерполяция признаков с нашей универсальной функцией
        new_features_x2 = knn_interpolate_robust(
            x=features_flat,
            pos_x=xyz_flat,
            pos_y=xyz_x2_flat,
            k=self.k,
            batch_x=batch_old,
            batch_y=batch_new
        )
        
        # Преобразуем обратно в нужный формат
        new_features_x2 = new_features_x2.view(B, N_x2, -1).transpose(1, 2)  # (B, C, N_x2)
        
        # 4. Второй уровень апсемплинга (x2 → x4)
        xyz_x4, features_2 = self.unpool_2(xyz_x2, new_features_x2)  # (B, 3, 4*N), (B, C, 2*N)
        
        return xyz_x4


# Для обратной совместимости
Generator = TransformerGenerator


if __name__ == '__main__':
    # Тестирование
    cfg = {'k': 8, 'feat_dim': 128, 'res_conv_dim': 128}
    
    # Создаем модель
    model = TransformerGenerator(cfg).cuda()
    
    # Тестовый вход
    xyz = torch.rand(8, 3, 1024).cuda()
    
    # Замеряем производительность
    import time
    
    # Прогреваем GPU
    for _ in range(5):
        with torch.no_grad():
            _ = model(xyz)
    
    # Засекаем время
    torch.cuda.synchronize()
    start_time = time.time()
    
    num_iterations = 20
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(xyz)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    print(f"Среднее время выполнения: {avg_time:.4f} сек")
    print(f"FPS: {1.0/avg_time:.2f}")
    
    # Проверяем размер выхода
    out = model(xyz)
    print(f"Размерность выхода: {out.shape}")  # Должно быть (8, 3, 4096)
    
    # Проверяем использование памяти
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    _ = model(xyz)
    memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"Максимальное использование памяти: {memory_usage:.2f} ГБ")