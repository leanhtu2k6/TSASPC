import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from sklearn.cluster import k_means, AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, pairwise_distances, \
    pairwise_distances_argmin
from sklearn.mixture import GaussianMixture
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


def plot_pseudo_labels(save_path, num_classes, *labels):
    num_pics = len(labels) + 1
    color_map = plt.get_cmap('turbo').copy()
    color_map.set_under('w')
    fig = plt.figure(figsize=(15, num_pics * 1.5))

    barprops = dict(aspect='auto', cmap=color_map, interpolation='nearest', vmin=0, vmax=num_classes - 1)

    # dic = {1:'(a)', 2:'(b)', 3:'(c)', 4:'(d)', 5:'(e)', 6:'(f)'}
    for i, label in enumerate(labels):
        plt.subplot(num_pics, 1, i + 1)
        plt.xticks([])
        plt.yticks([])
        # plt.ylabel(dic[i+1]+'      ', rotation = 0, size=20)
        plt.imshow([label], **barprops)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


def eval_pseudo_labels(pseudo_labels, gt_labels):
    true_num = np.sum(pseudo_labels == gt_labels)
    pseudo_num = len(gt_labels) - np.sum(pseudo_labels == -1)
    return true_num, pseudo_num


def intersection_labels(*labels):
    assert len(labels) >= 2
    out_labels = np.zeros_like(labels[0]) - 1
    out_labels[labels[0] == labels[1]] = labels[0][labels[0] == labels[1]]
    for i in range(2, len(labels)):
        out_labels[~(out_labels == labels[i])] = -1
    return out_labels


def k_shape_clustering(stamps, features, classes, n_clusters=None, random_state=42):
    """ k-Shape clustering for temporal pseudo label generation

    Args:
        stamps (array): an 1-D array containing all timestamp index
        features (array): features (L, D)
        classes (array): ground truth classes
        n_clusters (int, optional): number of clusters (default = len(stamps))
        random_state (int): random seed for reproducibility
    """
    if n_clusters is None:
        n_clusters = len(stamps)

    # Chuẩn hóa chuỗi thời gian
    scaler = TimeSeriesScalerMeanVariance()
    X_scaled = scaler.fit_transform(features)

    # Áp dụng thuật toán k-Shape
    ks = KShape(n_clusters=n_clusters, n_init=3, random_state=random_state)
    cluster_labels = ks.fit_predict(X_scaled)

    # Gán nhãn dựa trên cụm và timestamp
    label2class = dict()
    for i in range(n_clusters):
        # chọn frame đại diện của cụm gần nhất với stamp tương ứng
        cluster_indices = np.where(cluster_labels == i)[0]
        if len(cluster_indices) == 0:
            continue
        # tìm stamp gần nhất với tâm cụm
        center = ks.cluster_centers_[i].ravel()
        min_dist = float('inf')
        closest_stamp = stamps[0]
        for s in stamps:
            if s < len(features):
                dist = np.linalg.norm(features[s] - center)
                if dist < min_dist:
                    min_dist = dist
                    closest_stamp = s
        label2class[i] = classes[closest_stamp]

    # Sinh nhãn giả dựa trên cụm
    output_classes = np.zeros_like(classes) - 1
    for i, label in enumerate(cluster_labels):
        output_classes[i] = label2class.get(label, -1)

    true_num, pseudo_num = eval_pseudo_labels(output_classes, classes)

    for each in stamps:
        assert classes[each] == output_classes[each]

    return output_classes, true_num, pseudo_num, 0


def dp_means(stamps, features, classes, lam=3.5, max_iter=100):
    """
    DP-means clustering for pseudo label generation
    Args:
        stamps (array): chỉ số timestamp (khung có nhãn thật)
        features (array): đặc trưng (L, D)
        classes (array): nhãn thật (L,)
        lam (float): penalty tạo cụm mới
        max_iter (int): số vòng lặp tối đa
    """
    n = len(features)
    k = 1
    mu = [np.mean(features, axis=0)]  # khởi tạo tâm đầu tiên
    z = np.zeros(n, dtype=int)  # mỗi điểm ban đầu thuộc cụm

    iter_count = 0

    for _ in range(max_iter):
        changed = False

        # --- Gán cụm ---
        for i in range(n):
            dists = [np.linalg.norm(features[i] - mu_c) ** 2 for mu_c in mu]
            min_dist = np.min(dists)
            if min_dist > lam:  # tạo cụm mới
                mu.append(features[i])
                z[i] = k
                k += 1
                changed = True
            else:
                new_label = np.argmin(dists)
                if z[i] != new_label:
                    z[i] = new_label
                    changed = True

        # --- Cập nhật tâm cụm ---
        for c in range(k):
            members = features[z == c]
            if len(members) > 0:
                mu[c] = np.mean(members, axis=0)

        if not changed:
            break

    print(f"[DP-means] Converged after {iter_count} iterations, total clusters: {k}")

    # --- Sinh nhãn giả ---
    output_classes = np.zeros_like(classes) - 1
    label2class = {}

    for j in range(k):
        # Tìm timestamp gần nhất trong cụm
        indices = np.where(z == j)[0]
        if len(indices) == 0:
            continue
        closest_stamp = min(stamps, key=lambda s: np.min(np.abs(indices - s)))
        label2class[j] = classes[closest_stamp]
        for idx in indices:
            output_classes[idx] = label2class[j]

    true_num, pseudo_num = eval_pseudo_labels(output_classes, classes)
    for each in stamps:
        assert classes[each] == output_classes[each]

    return output_classes, true_num, pseudo_num, 0


def hdp(stamps, features, classes, lam_local=3.0, lam_global=6.0, max_iter=100):
    """
    Hard Gaussian HDP clustering for pseudo label generation

    Args:
        stamps (array): chỉ số timestamp (frame có nhãn thật)
        features (array): đặc trưng (L, D)
        classes (array): nhãn thật (L,)
        lam_local (float): penalty khi tạo local cluster mới
        lam_global (float): penalty khi tạo global cluster mới
        max_iter (int): số vòng lặp tối đa
    Returns:
        output_classes, true_num, pseudo_num, stop_iter
    """
    # số lượng dataset (ở đây mỗi video coi như 1 dataset)
    # để giữ cấu trúc code, ta coi toàn bộ video là 1 dataset duy nhất
    n = len(features)
    j = 0  # chỉ 1 dataset
    g = 1  # số global cluster
    k_j = 1  # số local cluster cho dataset j
    mu = [np.mean(features, axis=0)]  # global mean ban đầu

    # khởi tạo chỉ số local và mapping local->global
    z_ij = np.ones(n, dtype=int)
    v_jc = {1: 1}

    for iteration in range(max_iter):
        changed = False

        # ----- STEP 4: cập nhật gán điểm -----
        for i in range(n):
            # khoảng cách tới từng global cluster
            d_ijp = [np.linalg.norm(features[i] - mu_p) ** 2 for mu_p in mu]

            # cộng penalty λ_local cho global cluster chưa có trong dataset j
            for p in range(g):
                if p + 1 not in v_jc.values():
                    d_ijp[p] += lam_local

            # kiểm tra có cần tạo cluster mới không
            min_dist = np.min(d_ijp)
            if min_dist > lam_local + lam_global:
                # tạo local + global mới
                k_j += 1
                z_ij[i] = k_j
                g += 1
                mu.append(features[i])
                v_jc[k_j] = g
                changed = True
            else:
                p_hat = np.argmin(d_ijp) + 1
                found = False
                for c, vp in v_jc.items():
                    if vp == p_hat:
                        z_ij[i] = c
                        found = True
                        break
                if not found:
                    k_j += 1
                    z_ij[i] = k_j
                    v_jc[k_j] = p_hat
                    changed = True

        # ----- STEP 5: cập nhật local clusters -----
        for c in range(1, k_j + 1):
            S_jc = features[z_ij == c]
            if len(S_jc) == 0:
                continue
            mu_bar_jc = np.mean(S_jc, axis=0)

            d_bar_jcp = [np.sum(np.linalg.norm(S_jc - mu[p], axis=1) ** 2) for p in range(g)]
            internal_error = np.sum(np.linalg.norm(S_jc - mu_bar_jc, axis=1) ** 2)

            if np.min(d_bar_jcp) > lam_global + internal_error:
                g += 1
                v_jc[c] = g
                mu.append(mu_bar_jc)
                changed = True
            else:
                v_jc[c] = np.argmin(d_bar_jcp) + 1

        # ----- STEP 6: cập nhật lại mean global -----
        new_mu = []
        for p in range(1, g + 1):
            points = []
            for c, vp in v_jc.items():
                if vp == p:
                    points.extend(features[z_ij == c])
            if len(points) > 0:
                new_mu.append(np.mean(points, axis=0))
            else:
                new_mu.append(mu[p - 1])
        mu = new_mu

        if not changed:
            print(f"[HDP] Converged after {iteration + 1} iterations, total global={g}, local={k_j}")
            break

    # ----- Sinh nhãn giả -----
    output_classes = np.zeros_like(classes) - 1
    label2class = {}

    for c, p in v_jc.items():
        # tìm timestamp gần nhất trong local cluster
        indices = np.where(z_ij == c)[0]
        if len(indices) == 0:
            continue
        closest_stamp = min(stamps, key=lambda s: np.min(np.abs(indices - s)))
        label2class[c] = classes[closest_stamp]
        for idx in indices:
            output_classes[idx] = label2class[c]

    true_num, pseudo_num = eval_pseudo_labels(output_classes, classes)
    for each in stamps:
        assert classes[each] == output_classes[each]

    return output_classes, true_num, pseudo_num, 0


def temporal_agnes(stamps, features, classes, metric='euclidean', linkage='average'):
    """ temporal agnes

    Args:
        stamps (array): an 1-D array containing all timestamp index
        features (array): features
        classes (array): classes
        metric (str, optional): ['euclidean', 'cosine', 'seuclidean']. Defaults to 'euclidean'.
        linkage (str, optional): ['average', 'max']. Defaults to 'average'.
    """
    n = len(stamps)
    length = features.shape[0]

    dist_matrix = pairwise_distances(features, metric=metric)
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            else:
                dist_matrix[stamps[i], stamps[j]] = 1e9

    cluster_list = []
    for i in range(length - 1):
        cluster_list.append({'begin_index': i, 'end_index': i + 1, 'dist': dist_matrix[i, i + 1]})
    cluster_list.append({'begin_index': length - 1, 'end_index': length, 'dist': float('inf')})

    def update_dis_average(pre, cur, post, flag, linkage):
        pre_num = pre['end_index'] - pre['begin_index']
        cur_num = cur['end_index'] - cur['begin_index']
        post_num = post['end_index'] - post['begin_index']
        if linkage == 'average':
            if flag:
                pre_dist = pre['dist']
                new_dist = np.sum(
                    dist_matrix[pre['begin_index']:pre['end_index'], post['begin_index']:post['end_index']])
                new_dist = (pre_dist * pre_num * cur_num + new_dist) / float(pre_num * (cur_num + post_num))
            else:
                cur_dist = cur['dist']
                new_dist = np.sum(
                    dist_matrix[pre['begin_index']:pre['end_index'], post['begin_index']:post['end_index']])
                new_dist = (cur_dist * cur_num * post_num + new_dist) / float(post_num * (cur_num + pre_num))
        else:
            if flag:
                pre_dist = pre['dist']
                new_dist = np.max(
                    dist_matrix[pre['begin_index']:pre['end_index'], post['begin_index']:post['end_index']])
                new_dist = max(new_dist, pre_dist)
            else:
                cur_dist = cur['dist']
                new_dist = np.max(
                    dist_matrix[pre['begin_index']:pre['end_index'], post['begin_index']:post['end_index']])
                new_dist = max(new_dist, cur_dist)
        return new_dist

    cur_cluster_num = length
    while cur_cluster_num > n:
        # find min distance
        tmp_dist = float('inf')
        tmp_min_index = 0
        for i, each in enumerate(cluster_list):
            if each['dist'] < tmp_dist:
                tmp_min_index = i
                tmp_dist = each['dist']

        # update distances
        if tmp_min_index > 0:
            cluster_list[tmp_min_index - 1]['dist'] = update_dis_average(cluster_list[tmp_min_index - 1],
                                                                         cluster_list[tmp_min_index],
                                                                         cluster_list[tmp_min_index + 1], True, linkage)
        if tmp_min_index < len(cluster_list) - 2:
            cluster_list[tmp_min_index]['dist'] = update_dis_average(cluster_list[tmp_min_index],
                                                                     cluster_list[tmp_min_index + 1],
                                                                     cluster_list[tmp_min_index + 2], False, linkage)
            # print(cluster_list[tmp_min_index]['dist'])
        if tmp_min_index == len(cluster_list) - 2:
            cluster_list[tmp_min_index]['dist'] = float('inf')

        # update clusters
        cluster_list[tmp_min_index]['end_index'] = cluster_list[tmp_min_index + 1]['end_index']
        del cluster_list[tmp_min_index + 1]
        cur_cluster_num -= 1

    output_classes = np.zeros_like(classes) - 1
    for i in range(n):
        output_classes[cluster_list[i]['begin_index']:cluster_list[i]['end_index']] = classes[stamps[i]]

    true_num, pseudo_num = eval_pseudo_labels(output_classes, classes)

    for each in stamps:
        assert classes[each] == output_classes[each]

    return output_classes, true_num, pseudo_num, 0


# useless function
def agglomerative_clustering(stamps, features, classes, metric='euclidean', tol=1e-4):
    """ agglomerative clustering (useless function)

    Args:
        stamps (array): an 1-D array containing all timestamp index
        features (array): features
        classes (array): classes
        metric (str, optional): ['euclidean', 'cosine', 'seuclidean']. Defaults to 'euclidean'.
    """
    n = len(stamps)
    length = features.shape[0]
    dist_matrix = pairwise_distances(features, metric=metric)

    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            else:
                dist_matrix[stamps[i], stamps[j]] = 1e9

    connectivity = diags([1, 1, 1], [-1, 0, 1], shape=(length, length))
    connectivity = connectivity.tolil()
    connectivity[stamps[0], stamps[1]] = 1
    for i in range(1, n - 1):
        connectivity[stamps[i], stamps[i - 1]] = 1
        connectivity[stamps[i], stamps[i + 1]] = 1
    connectivity[stamps[-1], stamps[-2]] = 1

    model = AgglomerativeClustering(n_clusters=n, metric='precomputed', connectivity=connectivity, linkage='average',
                                    compute_distances=True).fit(dist_matrix)
    # model = AgglomerativeClustering(n_clusters=n, connectivity=connectivity, linkage='ward').fit(features)

    label2class = dict()
    for i in range(n):
        label_key = model.labels_[stamps[i]]
        class_value = classes[stamps[i]]
        label2class[label_key] = class_value

    output_classes = np.zeros_like(classes) - 1
    for i in range(len(classes)):
        output_classes[i] = label2class.get(model.labels_[i], -1)

    true_num, pseudo_num = eval_pseudo_labels(output_classes, classes)

    return output_classes, true_num, pseudo_num, 0


def constrained_k_medoids(stamps, features, classes, metric='euclidean', tol=1e-4):
    """ constrained K medoids

    Args:
        stamps (array): an 1-D array containing all timestamp index
        features (array): features
        classes (array): classes
        metric (str, optional): ['euclidean', 'cosine', 'seuclidean']. Defaults to 'euclidean'.
    """
    n = len(stamps)
    length = features.shape[0]
    medoids = stamps.copy()
    dist_matrix = pairwise_distances(features, metric=metric)

    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            else:
                dist_matrix[stamps[i], stamps[j]] = 1e9

    max_iter = 300
    iter = 0
    flag = True
    while iter < max_iter and flag:
        flag = False
        # find boundary
        boundary = []
        boundary.append(0)
        for i in range(n - 1):
            tmp_dist_sum = float('inf')
            tmp_index = stamps[i]
            for l in range(stamps[i], stamps[i + 1]):
                dist_sum = dist_matrix[medoids[i], stamps[i]:l + 1].sum() + dist_matrix[
                    medoids[i + 1], l + 1:stamps[i + 1] + 1].sum()
                if dist_sum < tmp_dist_sum:
                    tmp_dist_sum = dist_sum
                    tmp_index = l
            boundary.append(tmp_index + 1)
        boundary.append(len(classes))

        # find new medoids
        for i in range(n):
            tmp_index = medoids[i]
            tmp_dist_sum = dist_matrix[tmp_index, boundary[i]:boundary[i + 1]].sum()
            for l in range(boundary[i], boundary[i + 1]):
                dist_sum = dist_matrix[l, boundary[i]:boundary[i + 1]].sum()
                if dist_sum < tmp_dist_sum - tol:
                    flag = True
                    tmp_dist_sum = dist_sum
                    tmp_index = l
            medoids[i] = tmp_index

        iter += 1

    output_classes = np.zeros_like(classes) - 1
    for i in range(n):
        output_classes[boundary[i]:boundary[i + 1]] = classes[stamps[i]]

    true_num, pseudo_num = eval_pseudo_labels(output_classes, classes)

    for each in stamps:
        assert classes[each] == output_classes[each]

    return output_classes, true_num, pseudo_num, iter


def energy_function(stamps, features, classes, metric='euclidean'):
    # only support euclidean distance
    n = len(stamps)
    length = features.shape[0]
    output_classes = np.zeros_like(classes) - 1

    output_classes[:stamps[0]] = classes[stamps[0]]  # frames before first single frame has same label

    # Forward to find action boundaries
    left_bound = [0]
    for i in range(n - 1):
        start = stamps[i]
        end = stamps[i + 1] + 1
        left_score = np.zeros(end - start - 1)
        for t in range(start + 1, end):
            center_left = np.mean(features[left_bound[-1]:t, :], axis=0)
            diff_left = features[start:t, :] - center_left.reshape(1, -1)
            score_left = np.linalg.norm(diff_left, axis=1).mean()

            center_right = np.mean(features[t:end, :], axis=0)
            diff_right = features[t:end, :] - center_right.reshape(1, -1)
            score_right = np.linalg.norm(diff_right, axis=1).mean()

            left_score[t - start - 1] = ((t - start) * score_left + (end - t) * score_right) / (end - start)

        cur_bound = np.argmin(left_score) + start + 1
        left_bound.append(cur_bound)

    # Backward to find action boundaries
    right_bound = [length]
    for i in range(n - 1, 0, -1):
        start = stamps[i - 1]
        end = stamps[i] + 1
        right_score = np.zeros(end - start - 1)
        for t in range(end - 1, start, -1):
            center_left = np.mean(features[start:t, :], axis=0)
            diff_left = features[start:t, :] - center_left.reshape(1, -1)
            score_left = np.linalg.norm(diff_left, axis=1).mean()

            center_right = np.mean(features[t:right_bound[-1], :], axis=0)
            diff_right = features[t:end, :] - center_right.reshape(1, -1)
            score_right = np.linalg.norm(diff_right, axis=1).mean()

            right_score[t - start - 1] = ((t - start) * score_left + (end - t) * score_right) / (end - start)

        cur_bound = np.argmin(right_score) + start + 1
        right_bound.append(cur_bound)

    # Average two action boundaries for same segment and generate pseudo labels
    left_bound = left_bound[1:]
    right_bound = right_bound[1:]
    num_bound = len(left_bound)
    for i in range(num_bound):
        temp_left = left_bound[i]
        temp_right = right_bound[num_bound - i - 1]
        middle_bound = int((temp_left + temp_right) / 2)
        output_classes[stamps[i]:middle_bound] = classes[stamps[i]]
        output_classes[middle_bound:stamps[i + 1] + 1] = classes[stamps[i + 1]]

    output_classes[stamps[-1]:] = classes[stamps[-1]]  # frames after last single frame has same label

    true_num, pseudo_num = eval_pseudo_labels(output_classes, classes)

    for each in stamps:
        assert classes[each] == output_classes[each]

    return output_classes, true_num, pseudo_num, 0


def ensemble(stamps, features, classes, metric='euclidean'):
    output_classes1, _, _, _ = energy_function(stamps, features, classes)
    output_classes2, _, _, _ = constrained_k_medoids(stamps, features, classes, metric)
    output_classes3, _, _, _ = agglomerative_clustering(stamps, features, classes, metric)
    output_classes = intersection_labels(output_classes1, output_classes2, output_classes3)

    true_num, pseudo_num = eval_pseudo_labels(output_classes, classes)

    return output_classes, true_num, pseudo_num, 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="50salads", help='three dataset: breakfast, 50salads, gtea')
    parser.add_argument('--metric', default="euclidean", help='three metrics: euclidean, cosine, seuclidean')
    parser.add_argument('--feature', default="1024", help='1024 or 2048 or all')
    parser.add_argument('--type', default="all",
                        help='all, energy_function, constrained_k_medoids, temporal_agnes, dp_means, hdp, k_shape_clustering')

    args = parser.parse_args()

    dataset_name = args.dataset
    sample_rate = 1
    if dataset_name == "50salads":
        sample_rate = 2

    if args.feature == "1024":
        pseudo_label_dir = "data/I3D_1024/" + dataset_name + "/"
    elif args.feature == "2048":
        pseudo_label_dir = "data/I3D_2048/" + dataset_name + "/"
    else:
        pseudo_label_dir = "data/I3D_all/" + dataset_name + "/"

    if not os.path.exists(pseudo_label_dir):
        os.makedirs(pseudo_label_dir)

    # read action dict
    file_ptr = open("data/" + dataset_name + "/mapping.txt", 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    reverse_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
        reverse_dict[int(a.split()[0])] = a.split()[1]
    reverse_dict[-1] = 'no_label'

    # read timestamp index
    random_index = np.load("data/" + dataset_name + "_annotation_all.npy", allow_pickle=True).item()

    if args.type == 'all':
        total_true = [0] * 7
        total_pseudo = [0] * 7
        total_length = 0
    else:
        total_true = 0
        total_pseudo = 0
        total_length = 0

    # process each video
    for vid, stamp in random_index.items():
        # read features
        features = np.load("data/" + dataset_name + "/features/" + vid.split('.')[0] + '.npy')  # (D, L)
        if args.feature == "1024":
            features = features[:1024, ::sample_rate]
        elif args.feature == "2048":
            features = features[1024:, ::sample_rate]
        else:
            features = features[:, ::sample_rate]

        features = features.T  # (L, D)

        # read labels
        file_ptr = open("data/" + dataset_name + "/groundTruth/" + vid, 'r')
        content = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        classes = np.zeros(len(content))
        for i in range(len(classes)):
            classes[i] = actions_dict[content[i]]
        classes = classes[::sample_rate]

        if args.type == 'all':
            output_classes1, true_num1, pseudo_num1, _ = energy_function(stamp, features, classes)
            output_classes2, true_num2, pseudo_num2, _ = constrained_k_medoids(stamp, features, classes,
                                                                               metric=args.metric)
            output_classes3, true_num3, pseudo_num3, _ = temporal_agnes(stamp, features, classes, metric=args.metric)
            # output_classes_add, _, _, _ = temporal_agnes(stamp, features, classes, metric=args.metric, linkage='max')

            output_classes4 = intersection_labels(output_classes1, output_classes2)
            true_num4, pseudo_num4 = eval_pseudo_labels(output_classes4, classes)

            output_classes5 = intersection_labels(output_classes1, output_classes3)
            true_num5, pseudo_num5 = eval_pseudo_labels(output_classes5, classes)

            output_classes6 = intersection_labels(output_classes2, output_classes3)
            true_num6, pseudo_num6 = eval_pseudo_labels(output_classes6, classes)

            output_classes = intersection_labels(output_classes1, output_classes2,
                                                 output_classes3)  # , output_classes_add)
            true_num, pseudo_num = eval_pseudo_labels(output_classes, classes)

            print(vid.split('.')[0] + "    true num: {}, pseudo labels num: {}, length: {}, stop iter: {}".format(
                true_num, pseudo_num, len(classes), 0))

            true_num_list = [true_num1, true_num2, true_num3, true_num4, true_num5, true_num6, true_num]
            pseudo_num_list = [pseudo_num1, pseudo_num2, pseudo_num3, pseudo_num4, pseudo_num5, pseudo_num6, pseudo_num]
            for i in range(7):
                total_true[i] += true_num_list[i]
                total_pseudo[i] += pseudo_num_list[i]
            total_length += len(classes)

        elif args.type == 'temporal_agnes':
            output_classes, true_num, pseudo_num, _ = temporal_agnes(stamp, features, classes, metric='euclidean')
            print(vid.split('.')[0] + "    true num: {}, pseudo labels num: {}, length: {}, stop iter: {}".format(
                true_num, pseudo_num, len(classes), 0))
            total_true += true_num
            total_pseudo += pseudo_num
            total_length += len(classes)

        elif args.type == 'energy_function':
            output_classes, true_num, pseudo_num, _ = energy_function(stamp, features, classes)
            print(vid.split('.')[0] + "    true num: {}, pseudo labels num: {}, length: {}, stop iter: {}".format(
                true_num, pseudo_num, len(classes), 0))
            total_true += true_num
            total_pseudo += pseudo_num
            total_length += len(classes)

        elif args.type == 'constrained_k_medoids':
            output_classes, true_num, pseudo_num, _ = constrained_k_medoids(stamp, features, classes,
                                                                            metric=args.metric)
            print(vid.split('.')[0] + "    true num: {}, pseudo labels num: {}, length: {}, stop iter: {}".format(
                true_num, pseudo_num, len(classes), 0))
            total_true += true_num
            total_pseudo += pseudo_num
            total_length += len(classes)

        elif args.type == 'k_shape_clustering':
            output_classes, true_num, pseudo_num, _ = k_shape_clustering(stamp, features, classes)
            print(vid.split('.')[0] + "    true num: {}, pseudo labels num: {}, length: {}, stop iter: {}".format(
                true_num, pseudo_num, len(classes), 0))
            total_true += true_num
            total_pseudo += pseudo_num
            total_length += len(classes)

        elif args.type == 'dp_means':
            output_classes, true_num, pseudo_num, _ = dp_means(stamp, features, classes)
            print(
                f"{vid.split('.')[0]}    true num: {true_num}, pseudo labels num: {pseudo_num}, length: {len(classes)}, stop iter: 0")
            total_true += true_num
            total_pseudo += pseudo_num
            total_length += len(classes)

        elif args.type == 'hdp':
            output_classes, true_num, pseudo_num, _ = hdp(stamp, features, classes)
            print(f"{vid.split('.')[0]} true num: {true_num}, pseudo labels num: {pseudo_num}, length: {len(classes)}")
            total_true += true_num
            total_pseudo += pseudo_num
            total_length += len(classes)

        # save pseudo label
        file_ptr = open(pseudo_label_dir + vid, 'w')
        for each in output_classes:
            file_ptr.write(reverse_dict[each] + '\n')
        file_ptr.close()

        plot_pseudo_labels(pseudo_label_dir + vid.split('.')[0] + '.pdf', len(actions_dict), classes, output_classes)
        # plot_pseudo_labels(pseudo_label_dir+vid.split('.')[0]+'.pdf', len(actions_dict), classes, ts_only(stamp,features,classes)[0], output_classes1, output_classes2, output_classes3, output_classes)

    if args.type == 'all':
        for i in range(7):
            print("i: {}".format(i + 1))
            print("label rate: {}".format(total_pseudo[i] / float(total_length)))
            print("label acc: {}".format(total_true[i] / float(total_pseudo[i])))
    else:
        print("label rate: {}".format(total_pseudo / float(total_length)))
        print("label acc: {}".format(total_true / float(total_pseudo)))