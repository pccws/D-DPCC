import pandas as pd
import altair as alt

def reset_points_number():
    csv_file = "./results/old_results.csv"
    df = pd.read_csv(csv_file)
    my_map = {}
    import dataset_kitti as kt
    import tqdm
    dataset = kt.KittiDataset("/home/yelanggao/Dataset/kitti_odometry/dataset", sequences=17)
    for i in tqdm.tqdm(range(101)):
        points, remissions = dataset[i]
        my_map[i] = points.shape[0]
    df["ori_points_number"] = df["frame_index"].map(my_map)
    df["old_bpp"] = df["bpp"]
    df["bpp"] = df["bits"] / df["ori_points_number"]
    df.to_csv("./results/results.csv", index=False)
    

def plot_rd():
    csv_file = "./results/results.csv"
    df = pd.read_csv(csv_file)
    df = df[df["q_level"] == 11]
    df = df.drop(columns=["frame_index"])
    df = df.groupby(["ckpt", "sequence", "q_level"]).mean().reset_index()
    print(df)
    df.to_csv("./results/results_rd.csv")
    alt.Chart(df).mark_line(point=True).encode(
        x="bpp",
        y=alt.Y("d1_psnr"), #.scale(zero=False),
        # color=alt.Color("q_level:N"),
    ).save("./results/plot.png", scale_factor=2)

def plot_frame_vs_bpp_psnr():
    csv_file = "./results/results.csv"
    df = pd.read_csv(csv_file)
    df = df[df["sequence"] == 17]
    df = df[df["ckpt"] == "r3.pth"]
    df = df[df["q_level"] == 10]
    print(df)
    alt.Chart(df).mark_line(point=True).encode(
        x="frame_index",
        y=alt.Y("bpp").scale(zero=False),
        color=alt.Color("q_level:N"),
    ).save("./results/frame_index_bpp.png", scale_factor=2.5)

    alt.Chart(df).mark_line(point=True).encode(
        x="frame_index",
        y=alt.Y("d1_psnr").scale(zero=False),
        color=alt.Color("q_level:N"),
    ).save("./results/frame_index_psnr.png", scale_factor=2.5)

    df["psnr_vs_bpp"] = df["d1_psnr"] / df["bpp"]
    alt.Chart(df).mark_line(point=True).encode(
        x="frame_index",
        y=alt.Y("psnr_vs_bpp").scale(zero=False),
        color=alt.Color("q_level:N"),
    ).save("./results/frame_index_psnr_vs_bpp.png", scale_factor=2.5)

    df["rolling_psnr_vs_bpp"] = df["psnr_vs_bpp"].rolling(10, center=True, min_periods=5).mean()
    alt.Chart(df).mark_line(point=True).encode(
        x="frame_index",
        y=alt.Y("rolling_psnr_vs_bpp").scale(zero=False),
        color=alt.Color("q_level:N"),
    ).save("./results/frame_index_rolling_psnr_vs_bpp.png", scale_factor=2.5)

    df["rolling_bpp"] = df["bpp"].rolling(10, center=True, min_periods=5).mean()
    alt.Chart(df).mark_line(point=True).encode(
        x="frame_index",
        y=alt.Y("rolling_bpp").scale(zero=False),
        color=alt.Color("q_level:N"),
    ).save("./results/frame_index_rolling_bpp.png", scale_factor=2.5)

    df["rolling_d1psnr"] = df["d1_psnr"].rolling(10, center=True, min_periods=5).mean()
    alt.Chart(df).mark_line(point=True).encode(
        x="frame_index",
        y=alt.Y("rolling_d1psnr").scale(zero=False),
        color=alt.Color("q_level:N"),
    ).save("./results/frame_index_rolling_d1psnr.png", scale_factor=2.5)
    
def generate_kitti_ply():
    import dataset_kitti as kt
    import pyntcloud
    import tqdm
    dataset = kt.KittiDataset("/home/yelanggao/Dataset/kitti_odometry/dataset", sequences=17)
    for i in tqdm.tqdm(range(100)):
        points, remissions = dataset[i]
        # save points with pyntcloud
        cloud = pyntcloud.PyntCloud(pd.DataFrame(points, columns=["x", "y", "z"]))
        cloud.to_file(f"./results/kitti_ply/{i}.ply")


def main():
    plot_rd()
    plot_frame_vs_bpp_psnr()
    # generate_kitti_ply()
    # reset_points_number()


if __name__ == "__main__":
    main()
