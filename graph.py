import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph, GCNConv
from torch_scatter import scatter_max, scatter_mean
from torch_sparse import SparseTensor

# Для совместимости с существующим Discriminator.py
def init_weight_(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class FeatureNetGCN(nn.Module):
    def __init__(self, k=8, dim=128):
        super().__init__()
        self.k = k
        self.conv1 = GCNConv(3, dim)
        self.conv2 = GCNConv(dim, dim)
        self.conv3 = GCNConv(dim, dim)
        self.apply(init_weight_)

    def forward(self, x):
        """
        Вход:  (B, 3, N)
        Выход: (B, C, N), где C = dim
        """
        B, _, N = x.shape
        
        # Подготовка данных для PyG - сразу преобразуем в правильный формат
        x_flat = x.transpose(1, 2).reshape(-1, 3)  # (B*N, 3)
        batch = torch.arange(B, device=x.device).repeat_interleave(N)  # (B*N,)
        
        # Обработка через 3 GCN слоя
        h = x_flat
        for conv in [self.conv1, self.conv2, self.conv3]:
            # Строим граф kNN и применяем свертку
            edge_index = knn_graph(h, k=self.k, batch=batch)
            h = F.relu(conv(h, edge_index))
            
            # Применяем максимизационный пулинг по соседям
            row, col = edge_index
            dim_size = h.size(0)
            neighbor_features = h[col]
            h, _ = scatter_max(neighbor_features, row, dim=0, dim_size=dim_size)
        
        # Возврат к формату (B, C, N)
        return h.view(B, N, -1).transpose(1, 2)


# Класс-обертка для обратной совместимости
class FeatureNet(FeatureNetGCN):
    def __init__(self, k=8, dim=128, num_block=3):
        super().__init__(k=k, dim=dim)


class GraphResBlock(nn.Module):
    """Оптимизированный блок с графовыми свертками и остаточным соединением."""
    def __init__(self, in_dim, dim, k=8):
        super().__init__()
        self.k = k
        
        # Два GCN слоя
        self.conv1 = GCNConv(in_dim, dim)
        self.conv2 = GCNConv(dim, dim)
        
        # Слой проекции для остаточного соединения при разных размерностях
        self.shortcut = nn.Identity() if in_dim == dim else nn.Linear(in_dim, dim)
        
        self.apply(init_weight_)
        
    def forward(self, x, batch):
        identity = x
        
        # Применяем первую свертку и нелинейность
        edge_index = knn_graph(x, k=self.k, batch=batch)
        x = F.relu(self.conv1(x, edge_index))
        
        # Применяем вторую свертку
        x = self.conv2(x, edge_index)
        
        # Добавляем остаточное соединение
        x = x + self.shortcut(identity)
        
        return x, edge_index


class ResGraphConvUnpool(nn.Module):
    """Оптимизированный модуль для апсемплинга облака точек с остаточными блоками."""
    def __init__(self, k=8, in_dim=128, dim=128, num_blocks=12):
        super().__init__()
        self.k = k
        
        # Создаем последовательность блоков
        self.blocks = nn.ModuleList([
            GraphResBlock(in_dim if i == 0 else dim, dim, k) 
            for i in range(num_blocks)
        ])
        
        # Единый прогнозатор смещений с нормализацией
        self.offset_predictor = nn.Sequential(
            nn.Linear(dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 6),  # 6 = 3 координаты * 2 новые точки
            nn.Tanh()          # Ограничиваем для стабильности (-1, 1)
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
        x_flat = xyz.transpose(1, 2).reshape(-1, 3)  # (B*N, 3)
        h = features.transpose(1, 2).reshape(-1, features.size(1))  # (B*N, C)
        batch = torch.arange(B, device=xyz.device).repeat_interleave(N)  # (B*N,)
        
        # Применяем последовательность блоков
        for block in self.blocks:
            h, edge_index = block(h, batch)
        
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
    
    # Преобразуем расстояния в веса: меньшие расстояния получают большие веса
    dist = 1.0 / (dist + 1e-5)
    
    # Нормализуем веса для каждой целевой точки
    norm = scatter_mean(dist, row, dim=0, dim_size=pos_y.size(0))
    norm = norm[row]
    weight = dist / norm
    
    # Интерполируем признаки с помощью взвешенного суммирования
    y = scatter_mean(x[col] * weight.view(-1, 1), row, dim=0, dim_size=pos_y.size(0))
    
    return y


class Generator(nn.Module):
    """Оптимизированный генератор для апсемплинга облака точек."""
    def __init__(self, cfg):
        super().__init__()
        self.k = cfg['k']
        self.feat_dim = cfg['feat_dim']
        self.res_conv_dim = cfg['res_conv_dim']
        
        # Сеть извлечения признаков
        self.featurenet = FeatureNetGCN(self.k, self.feat_dim)
        
        # Две сети апсемплинга для 4-кратного увеличения плотности
        self.res_unpool_1 = ResGraphConvUnpool(self.k, self.feat_dim, self.res_conv_dim)
        self.res_unpool_2 = ResGraphConvUnpool(self.k, self.res_conv_dim, self.res_conv_dim)
        
    @torch.cuda.amp.autocast()  # Включаем смешанную точность для ускорения
    def forward(self, xyz):
        """
        Вход:  xyz - (B, 3, N), координаты точек
        Выход: upsampled_xyz - (B, 3, 4*N), координаты после апсемплинга
        """
        # 1. Извлекаем признаки
        features = self.featurenet(xyz)  # (B, C, N)
        
        # 2. Первый уровень апсемплинга (x2)
        xyz_x2, features_1 = self.res_unpool_1(xyz, features)  # (B, 3, 2*N), (B, C, N)
        
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
        xyz_x4, features_2 = self.res_unpool_2(xyz_x2, new_features_x2)  # (B, 3, 4*N), (B, C, 2*N)
        
        return xyz_x4


if __name__ == '__main__':
    # Profile forward
    cfg = {'k': 8, 'feat_dim': 128, 'res_conv_dim': 128}
    
    # Создаем модель
    model = Generator(cfg).cuda()
    
    # Тестовый вход
    xyz = torch.rand(24, 3, 1024).cuda()
    
    # Замеряем производительность
    import time
    
    # Прогреваем GPU
    for _ in range(5):
        with torch.no_grad():
            _ = model(xyz)
    
    # Засекаем время
    torch.cuda.synchronize()
    start_time = time.time()
    
    num_iterations = 100
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
    print(f"Размерность выхода: {out.shape}")  # Должно быть (24, 3, 4096)
    
    # Проверяем использование памяти
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    _ = model(xyz)
    memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"Максимальное использование памяти: {memory_usage:.2f} ГБ")