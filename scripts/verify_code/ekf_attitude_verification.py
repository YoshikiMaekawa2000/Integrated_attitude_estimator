import yaml
import csv
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


class VerifyInfer:
    def __init__(self, CFG):
        self.CFG = CFG

        self.infer_log_path = CFG["infer_log_path"]

        self.csv_1 = CFG["csv_1"]
        self.csv_1_path = self.infer_log_path + self.csv_1
        self.csv_1_data, self.csv_1_MAE_roll, self.csv_1_MAE_pitch = self.get_csv_data(self.csv_1_path)


        self.csv_2 = CFG["csv_2"]
        self.csv_2_path = self.infer_log_path + self.csv_2
        self.csv_2_data, self.csv_2_MAE_roll, self.csv_2_MAE_pitch = self.get_csv_data(self.csv_2_path)

        self.csv_3 = CFG["csv_3"]
        self.csv_3_path = self.infer_log_path + self.csv_3
        self.csv_3_data, self.csv_3_MAE_roll, self.csv_3_MAE_pitch = self.get_csv_data(self.csv_3_path)

        self.fig_name = CFG["fig_name"]
        self.save_fig_path = self.infer_log_path + self.fig_name
        self.end_time = int(CFG['end_time'])

    def get_csv_data(self, path):
        data_list = []
        acc_roll_diff = 0.0
        acc_pitch_diff = 0.0

        with open(path) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                # in row
                # [time, roll, pitch, yaw, gt_roll, gt_pitch, gt_yaw]
                time = float(row[0])
                roll = float(row[1])*180/np.pi
                pitch = float(row[2])*180/np.pi
                yaw = float(row[3])*180/np.pi
                gt_roll = float(row[4])*180/np.pi
                gt_pitch = float(row[5])*180/np.pi
                gt_yaw = float(row[6])*180/np.pi

                diff_roll = abs(roll - gt_roll)
                diff_pitch = abs(pitch - gt_pitch)

                if diff_roll > 20.0 and time < 1.0:
                    continue
                if diff_pitch > 20.0 and time < 1.0:
                    continue

                acc_roll_diff += diff_roll
                acc_pitch_diff += diff_pitch

                tmp_row = [time, roll, pitch, yaw, gt_roll, gt_pitch, gt_yaw]

                data_list.append(tmp_row)

        mae_roll = acc_roll_diff / len(data_list)
        mae_pitch = acc_pitch_diff / len(data_list)

        return data_list, mae_roll, mae_pitch

    def get_data_list(self, data_list):
        time_list = []
        roll_list = []
        pitch_list = []
        gt_roll_list = []
        gt_pitch_list = []

        for row in data_list:
            tmp_time = float(row[0])
            tmp_roll = float(row[1])
            tmp_pitch = float(row[2])
            tmp_gt_roll = float(row[4])
            tmp_gt_pitch = float(row[5])

            time_list.append(tmp_time)
            roll_list.append(tmp_roll)
            pitch_list.append(tmp_pitch)
            gt_roll_list.append(tmp_gt_roll)
            gt_pitch_list.append(tmp_gt_pitch)

        return time_list, roll_list, pitch_list, gt_roll_list, gt_pitch_list

    def save_seq_graph(self):
        inference_time_list_1 = []
        inference_time_list_2 = []
        inference_time_list_3 = []
        csv_1_roll_list = []
        csv_1_pitch_list = []
        csv_2_roll_list = []
        csv_2_pitch_list = []
        csv_3_roll_list = []
        csv_3_pitch_list = []
        gt_roll_list = []
        gt_pitch_list = []

        inference_time_list_1, csv_1_roll_list, csv_1_pitch_list, gt_roll_list, gt_pitch_list = self.get_data_list(self.csv_1_data)
        inference_time_list_2, csv_2_roll_list, csv_2_pitch_list, gt_roll_list, gt_pitch_list = self.get_data_list(self.csv_2_data)
        inference_time_list_3, csv_3_roll_list, csv_3_pitch_list, gt_roll_list, gt_pitch_list = self.get_data_list(self.csv_3_data)

        print("Save Seq Graph")
        fig_1 = plt.figure(figsize=(19.20, 10.80))
        ax_roll = fig_1.add_subplot(2, 1, 1)
        ax_pitch = fig_1.add_subplot(2, 1, 2)

        c1,c2,c3,c4 = "blue","green","red","black" # All, W/O IMU angle, W/O IMU angle and Vel, GT

        ax_roll.plot(inference_time_list_3, gt_roll_list, c=c4, label="GT Roll")
        ax_roll.plot(inference_time_list_1, csv_1_roll_list, c=c1, label="DNN, Velocity, Gyro")
        ax_roll.plot(inference_time_list_2, csv_2_roll_list, c=c2, label="DNN, Velocity")
        ax_roll.plot(inference_time_list_3, csv_3_roll_list, c=c3, label="DNN")

        ax_pitch.plot(inference_time_list_3, gt_pitch_list, c=c4, label="GT Pitch")
        ax_pitch.plot(inference_time_list_1, csv_1_pitch_list, c=c1, label="DNN, Velocity, Gyro")
        ax_pitch.plot(inference_time_list_2, csv_2_pitch_list, c=c2, label="DNN, Velocity")
        ax_pitch.plot(inference_time_list_3, csv_3_pitch_list, c=c3, label="DNN")

        ax_roll.set_xlabel("Inference Time [sec]")
        ax_roll.set_ylabel("Roll [deg]")
        ax_roll.set_ylim(-30.0, 30.0)

        ax_pitch.set_xlabel("Inference Time [sec]")
        ax_pitch.set_ylabel("Pitch [deg]")
        ax_pitch.set_ylim(-30.0, 30.0)
        print("Last Inference Time: ", inference_time_list_3[-1], "[sec]")

        ax_roll.set_xlim(0, self.end_time)
        ax_pitch.set_xlim(0, self.end_time)

        ax_roll.legend()
        ax_pitch.legend()

        print("CSV 1 MAE Roll: ", self.csv_1_MAE_roll, "[deg]")
        print("CSV 1 MAE Pitch: ", self.csv_1_MAE_pitch, "[deg]")
        print("CSV 2 MAE Roll: ", self.csv_2_MAE_roll, "[deg]")
        print("CSV 2 MAE Pitch: ", self.csv_2_MAE_pitch, "[deg]")
        print("CSV 3 MAE Roll: ", self.csv_3_MAE_roll, "[deg]")
        print("CSV 3 MAE Pitch: ", self.csv_3_MAE_pitch, "[deg]")

        plt.suptitle(
            "MAE of Roll(DNN, Velocity, Gyro): {:.3f}".format(self.csv_1_MAE_roll) + " [deg], MAE of Pitch(DNN, Velocity, Gyro): {:.3f}".format(self.csv_1_MAE_pitch) + " [deg]" + "\n"
            "MAE of Roll(DNN, Velocity): {:.3f}".format(self.csv_2_MAE_roll) + " [deg], MAE of Pitch(DNN, Velocity): {:.3f}".format(self.csv_2_MAE_pitch) + " [deg]" + "\n"
            "MAE of Roll(DNN): {:.3f}".format(self.csv_3_MAE_roll) + " [deg], MAE of Pitch(DNN): {:.3f}".format(self.csv_3_MAE_pitch) + " [deg]"        
        , fontsize=20)

        plt.savefig(self.save_fig_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('./ekf_attitude_verification.py')

    parser.add_argument(
        '--verify_infer', '-vi',
        type=str,
        required=False,
        default='./ekf_attitude_verification.yaml',
    )

    FLAGS, unparsed = parser.parse_known_args()

    #Load yaml file
    try:
        print("Opening yaml file %s", FLAGS.verify_infer)
        CFG = yaml.safe_load(open(FLAGS.verify_infer, 'r'))
    except Exception as e:
        print(e)
        print("Error yaml file %s", FLAGS.verify_infer)
        quit()

    verify_infer = VerifyInfer(CFG)
    verify_infer.save_seq_graph()