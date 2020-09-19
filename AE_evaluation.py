import os
import cv2
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from AE_training import architecture_MVTEC


def calculate_TP_TN_FP_FN(ground_truth, predicted_mask):
    TP = np.sum(np.multiply((ground_truth == predicted_mask), predicted_mask == 1))
    TN = np.sum(np.multiply((ground_truth == predicted_mask), predicted_mask == 0))
    FP = np.sum(np.multiply((ground_truth != predicted_mask), predicted_mask == 1))
    FN = np.sum(np.multiply((ground_truth != predicted_mask), predicted_mask == 0))
    return TP, TN, FP, FN


def AP(TP, TN, FP, FN):
    return (TP+TN)/(TP+TN+FP+FN)


def DICE(TP, TN, FP, FN):
    if (2*TP+FP+FN) == 0:
        return 1
    else:
        return (2*TP)/(2*TP+FP+FN)


def FPR(TP, TN, FP, FN):
    return FP/(FP+TN)


def TPR(TP, TN, FP, FN):
    return TP/(TP+FN)


def Youden_statistic(TP, TN, FP, FN):
    if (TP+FN) == 0 or (TN+FP) == 0:
        return None
    else:
        return TP/(TP+FN) + TN/(TN+FP) - 1


def calculate_AUC(ROC_curve):
    AUC = 0
    FPR_values = [x[0] for x in ROC_curve]
    TPR_values = [x[1] for x in ROC_curve]
    for i in range(0, len(FPR_values)-1):
        a = TPR_values[i]
        b = TPR_values[i+1]
        h = FPR_values[i+1]-FPR_values[i]
        AUC += (a+b)*h/2
    return AUC


def plot_ROC_curve(metric_results):
    FPR_values = [x[1] for x in metric_results["FPR"]]
    TPR_values = [x[1] for x in metric_results["TPR"]]
    plt.plot(FPR_values, TPR_values, "-o")


def plot_YoundStat_thresh(metric_results):
    YoundStat_values = [x[1] for x in metric_results["YoundenStat"]]
    thresh_values = [x[0] for x in metric_results["YoundenStat"]]
    plt.plot(thresh_values, YoundStat_values, "-o")


def plot_DICE_thresh(metric_results):
    DICE_values = [x[1] for x in metric_results["DICE"]]
    thresh_values = [x[0] for x in metric_results["DICE"]]
    plt.plot(thresh_values, DICE_values, "-o")


def read_data(dataset):

    img_oryg_samples = []
    img_predict_samples = []
    ground_truth_samples = []

    for category in tqdm(os.listdir(f"data/{dataset}/test/")):
        if category == "good":
            continue
        for img_name in os.listdir(f"data/{dataset}/test/{category}/"):
            img_predict = cv2.imread(f"{dataset}/predicted/{category}/{img_name}", 0)

            img_oryg = cv2.imread(f"data/{dataset}/test/{category}/{img_name}", 0)
            img_oryg = cv2.resize(img_oryg, img_predict.shape)

        if dataset in ["wool_1", "wool_2"]:
            ground_truth = cv2.imread(f"data/{dataset}/ground_truth/{category}/{img_name}", 0)
        else:
            ground_truth = cv2.imread(f"data/{dataset}/ground_truth/{category}/{img_name[:-4]}_mask.png", 0)
        ground_truth = cv2.resize(ground_truth, img_predict.shape)
        ground_truth = (ground_truth > 0).astype(int)

        img_oryg_samples.append(img_oryg)
        img_predict_samples.append(img_predict)
        ground_truth_samples.append(ground_truth)

    return img_oryg_samples, img_predict_samples, ground_truth_samples


def calculate_loss(img_oryg_samples, img_predict_samples, win_size=11):

    loss_samples = []
    n = len(img_oryg_samples)
    for i in range(0, n):
        img_oryg = img_oryg_samples[i]
        img_predict = img_predict_samples[i]
        _, S = ssim(img_oryg, img_predict, gradient=False, full=True, multichannel=False, win_size=win_size)
        loss = 1/2 - S/2
        loss_samples.append(loss)

    return loss_samples


def calculate_metrics(loss_samples, ground_truth_samples, thresh_max=1):

    metric_results = dict()
    metric_results["DICE"] = []
    metric_results["YoundenStat"] = []
    metric_results["TPR"] = []
    metric_results["FPR"] = []

    thresh_min = 0
    thresh_values = np.linspace(thresh_min, np.sqrt(thresh_max), num=100)**2
    thresh_values = [x for x in thresh_values]

    for thresh in thresh_values:
        Dice_values = []
        YoundenStat_values = []
        TPR_values = []
        FPR_values = []
        n = len(loss_samples)
        for i in range(0, n):
            predicted_mask = (loss_samples[i] > thresh).astype(int)
            ground_truth = ground_truth_samples[i]
            TP, TN, FP, FN = calculate_TP_TN_FP_FN(ground_truth=ground_truth, predicted_mask=predicted_mask)
            Dice_values.append(DICE(TP, TN, FP, FN))
            YoundenStat_values.append(Youden_statistic(TP, TN, FP, FN))
            TPR_values.append(TPR(TP, TN, FP, FN))
            FPR_values.append(FPR(TP, TN, FP, FN))
        metric_results["DICE"].append((thresh, np.mean(Dice_values)))
        metric_results["YoundenStat"].append((thresh, np.mean(YoundenStat_values)))
        metric_results["TPR"].append((thresh, np.mean(TPR_values)))
        metric_results["FPR"].append((thresh, np.mean(FPR_values)))
    DICE_max_thresh, DICE_max = max(metric_results["DICE"], key=lambda x: x[1])
    YoundenStat_max_thresh, YoundenStat_max = max(metric_results["YoundenStat"], key=lambda x: x[1])
    ROC_curve = list(zip([x[1] for x in metric_results["FPR"]], [x[1] for x in metric_results["TPR"]]))
    ROC_curve = sorted(ROC_curve, key=lambda x: x[1])
    AUC = calculate_AUC(ROC_curve)

    return metric_results, DICE_max, YoundenStat_max, AUC


def create_predicted(dataset_name="carpet", latent_dim=100, training_loss="ssim", batch_size=8):

    autoencoder = architecture_MVTEC(input_shape=(128, 128, 1), latent_dim=latent_dim)
    path_to_load_model = f"model_weights/{dataset_name}/"
    name = f"a_{latent_dim}_loss_{training_loss}_batch_{batch_size}.hdf5"
    path_to_load_model += name
    autoencoder.load_weights(path_to_load_model)

    ROI_resized_size = (128, 128)

    for category in tqdm(os.listdir(f"data/{dataset_name}/test/")):

        try:
            os.makedirs(f"results/{dataset_name}/predicted/{category}/")
        except:
            pass

        try:
            os.makedirs(f"results/{dataset_name}/loss/{category}/")
        except:
            pass

        for img_name in tqdm(os.listdir(f"data/{dataset_name}/test/{category}/")):

            img_in = cv2.imread(f"data/{dataset_name}/test/{category}/{img_name}", 0)
            img_in = cv2.resize(img_in, ROI_resized_size)
            img_in = img_in.astype("float32") / 255.0
            X_test = []
            X_test.append(img_in)
            X_test = np.array(X_test)
            X_test = np.expand_dims(X_test, axis=-1)
            img_predict = autoencoder.predict(X_test)

            cv2.imwrite(f"results/{dataset_name}/predicted/{category}/{img_name}", img_predict[0, :, :, :]*255)

            mssim, grad, S = ssim(img_in[1:-1, 1:-1], img_predict[0, 1:-1, 1:-1, 0], gradient=True, full=True, multichannel=False)

            plt.clf()
            plt.imshow(1-S, vmax=1.5, cmap="jet")
            plt.colorbar()

            plt.savefig(f"results/{dataset_name}/loss/{category}/{img_name}")


if __name__ == "__main__":
    datasets = ["carpet"]
    for dataset in datasets:
        create_predicted(dataset_name=dataset)
        print("Koniec")
        img_oryg_samples, img_predict_samples, ground_truth_samples = read_data(dataset=dataset)
        loss_samples = calculate_loss(img_oryg_samples=img_oryg_samples, img_predict_sample=img_predict_samples, win_size=11)
        thresh_max = np.max(np.array(loss_samples))
        metric_results, DICE_max, YoundenStat_max, AUC = calculate_metrics(loss_samples=loss_samples, ground_truth_samples=ground_truth_samples, thresh_max=thresh_max)
