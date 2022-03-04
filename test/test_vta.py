import numpy as np
import json

from vcsl import build_vta_model

if __name__ == '__main__':
    uid = '0000ab50f69044d898ffd71a3d215a81-6445fe9aa1564a3783141ee9f8f56d3c'
    path = f'test/samples/{uid}.npy'
    sim = np.load(path)
    # sim = np.zeros((1,))

    gt = json.load(open('data/label_file_uuid_total.json'))
    print(gt[uid])

    dtw_model = build_vta_model("DTW", concurrency=4, )
    print(dtw_model.forward_sim([('DTW', sim)]))

    dp_model = build_vta_model("DP", concurrency=4)
    print(dp_model.forward_sim([('DP', sim)]))

    tn_model = build_vta_model("TN", concurrency=4)
    print(tn_model.forward_sim([('TN', sim)]))