def passenger_rfm(data_gmv, data_fin):
    #对数据进行padding后，sliding的方式计算rfm
    data_padding = np.zeros([data_fin.shape[0]+4, 2])
    data_padding[:2, 0] = data_fin[0]
    data_padding[2:data_fin.shape[0]+2, 0] = data_fin
    data_padding[-2:, 0] = data_fin[-1]
        
    data_padding[:2, 1] = data_gmv[0]
    data_padding[2:data_fin.shape[0]+2, 1] = data_gmv
    data_padding[-2:, 1] = data_gmv[-1]
    
    res = []
    for ii in range(data_fin.shape[0]):
        R = ii+1 if data_fin[ii] != 0 else (0 if ii == 0 else res[-3])
        F = np.sum(data_padding[ii:ii+4, 0])/4
        M = (np.sum(data_padding[ii:ii+4, 1]))/F*4 if F != 0 else -1
        res.extend([R, F, M])
    R_max, R_min, R_mean = np.max(res[0:len(res):3]), np.min(res[0:len(res):3]), np.mean(res[0:len(res):3])
    F_max, F_min, F_mean = np.max(res[1:len(res):3]), np.min(res[1:len(res):3]), np.mean(res[1:len(res):3])
    M_max, M_min = np.max(res[2:len(res):3]), np.min(res[2:len(res):3])
    M_data = res[2:len(res):3]
    count = 0
    s = 0
    for i in range(len(M_data)):
        if M_data[i]>0:
             s+=M_data[i]
    M_mean = s/count if count != 0 else 0

    return [R_max, R_min, R_mean, min(F_max, 70), min(F_min, 70), min(F_mean, 70), min(M_max, 500), min(M_min, 500), min(M_mean, 500), R_max-R_min]


def rfm_feature_table(data_gmv, data_finished):
    #根据完单量和gmv表，计算所有乘客的rfm特征
    feature_table = np.zeros([data_gmv.shape[0], 10])
    for i in range(data_gmv.shape[0]):
        feature_table[i, :] = passenger_rfm(data_gmv[i, :], data_finished[i, :])
    write_name = city_dir + "rfm_feature.tsv"
    pd_rfm = pd.DataFrame(feature_table, columns=["R_max", "R_min", "R_mean", "F_max", "F_min", "F_mean", "M_max", "M_min", "M_mean", "R_range"])
    print("rfm head is:", pd_rfm.head(5))
    pd_rfm.to_csv(write_name, sep='\t', index=False)
    print('rfm_feature save successyfully!')
    return feature_table
