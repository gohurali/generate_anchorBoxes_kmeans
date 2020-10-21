import yaml
import argparse
from kmeans import KMeans
from utils.data_utils import Dataset
from utils.data_utils import DataPrepper

config = yaml.safe_load(open("config.yaml"))
parser = argparse.ArgumentParser()
parser.add_argument('--k',type=int)
parser.add_argument("--fit-avg", type=bool, nargs='?',
                        const=True, default=False,
                        help="Run K-Means multiple times.")
parser.add_argument("--kmeans-iters",type=int,help="If fitting average, this is the number iterations k-means is run.")
parser.add_argument("--save-anchors", type=bool, nargs='?',
                        const=True, default=False,
                        help="Saves the anchors to current directory")
args = parser.parse_args()

def main():
    ds = Dataset(config)
    imgs, annots = ds.open_traffic_ds(config)
    dp = DataPrepper(x_data=imgs,y_data=annots)

    dp.x_data_scaled,dp.y_data_scaled = dp.rescale_data(dp.x_data,dp.y_data)
    km = KMeans(k=args.k,dataset=dp.y_data_scaled)
    if(args.fit_avg):
        km.fit_average(max_iterations=args.kmeans_iters)
        if(args.save_anchors):
            km.write_anchors(km.centroids)
    else:
        km.fit()
        if(args.save_anchors):
            km.write_anchors(km.centroids)

if __name__ == '__main__':
    main()