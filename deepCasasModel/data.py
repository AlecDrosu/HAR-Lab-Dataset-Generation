#!/usr/bin/env python3

import datetime
import os
import re
from collections import Counter
from datetime import datetime

import numpy as np
from keras.utils import pad_sequences

offset = 20
max_lenght = 2000

# cookActivities = {"milan": {"Other": offset,
#                             "Work": offset + 1,
#                             "Take_medicine": offset + 2,
#                             "Sleep": offset + 3,
#                             "Relax": offset + 4,
#                             "Leave_Home": offset + 5,
#                             "Eat": offset + 6,
#                             "Cook": offset + 7,
#                             "Bed_to_toilet": offset + 8,
#                             "Bathing": offset + 9,
#                             "Enter_home": offset + 10,
#                             "Personal_hygiene": offset + 11},
#                   }
# mappingActivities = {"milan": {"": "Other",
#                                "Master_Bedroom_Activity": "Other",
#                                "Meditate": "Other",
#                                "Chores": "Work",
#                                "Desk_Activity": "Work",
#                                "Morning_Meds": "Take_medicine",
#                                "Eve_Meds": "Take_medicine",
#                                "Sleep": "Sleep",
#                                "Read": "Relax",
#                                "Watch_TV": "Relax",
#                                "Leave_Home": "Leave_Home",
#                                "Dining_Rm_Activity": "Eat",
#                                "Kitchen_Activity": "Cook",
#                                "Bed_to_Toilet": "Bed_to_toilet",
#                                "Master_Bathroom": "Bathing",
#                                "Guest_Bathroom": "Bathing"},
#                      }

cookActivities = {"aruba": {"Other": offset,
                            "Meal_Preparation": offset + 1,
                            "Relax": offset + 2,
                            "Eating": offset + 3,
                            "Work": offset + 4,
                            "Sleeping": offset + 5,
                            "Wash_Dishes": offset + 6,
                            "Bed_to_Toilet": offset + 7,
                            "Enter_Home": offset + 8,
                            "Leave_Home": offset + 9,
                            "Housekeeping": offset + 10,
                            "Respirate": offset + 11},
                  }
mappingActivities = {"aruba": {"": "Other",
                               "Meal_Preparation": "Meal_Preparation",
                               "Relax": "Relax",
                               "Eating": "Eating",
                               "Work": "Work",
                               "Sleeping": "Sleeping",
                               "Wash_Dishes": "Wash_Dishes",
                               "Bed_to_Toilet": "Bed_to_Toilet",
                               "Enter_Home": "Enter_Home",
                               "Leave_Home": "Leave_Home",
                               "Housekeeping": "Housekeeping",
                               "Respirate": "Respirate"},
                     }

datasets = ["./dataset/aruba"]
datasetsNames = [i.split('/')[-1] for i in datasets]


def load_dataset(filename):
    # dateset fields
    timestamps = []
    sensors = []
    values = []
    activities = []

    activity = ''  # empty

    with open(filename, 'rb') as features:
        database = features.readlines()

        # start_line = int(len(database) * 0.5) # last 10 percent of the dataset
        for i, line in enumerate(database):  # each line
            f_info = line.decode().split()  # find fields
            try:
                if 'M' == f_info[2][0] or 'D' == f_info[2][0] or 'T' == f_info[2][0]:
                    # choose only M D T sensors, avoiding unexpected errors
                    if not ('.' in str(np.array(f_info[0])) + str(np.array(f_info[1]))):
                        f_info[1] = f_info[1] + '.000000'
                    timestamps.append(datetime.strptime(str(np.array(f_info[0])) + str(np.array(f_info[1])),
                                                        "%Y-%m-%d%H:%M:%S.%f"))
                    sensors.append(str(np.array(f_info[2])))
                    values.append(str(np.array(f_info[3])))

                    if len(f_info) == 4:  # if activity does not exist
                        activities.append(activity)
                    else:  # if activity exists
                        des = str(' '.join(np.array(f_info[4:])))
                        if 'begin' in des:
                            activity = re.sub('begin', '', des)
                            if activity[-1] == ' ':  # if white space at the end
                                activity = activity[:-1]  # delete white space
                            activities.append(activity)
                        if 'end' in des:
                            activities.append(activity)
                            activity = ''
            except IndexError:
                print(i, line)
    features.close()
    # dictionaries: assigning keys to values
    temperature = []
    for element in values:
        try:
            temperature.append(float(element))
        except ValueError:
            pass
    sensorsList = sorted(set(sensors))
    dictSensors = {}
    for i, sensor in enumerate(sensorsList):
        dictSensors[sensor] = i
    activityList = sorted(set(activities))
    dictActivities = {}
    for i, activity in enumerate(activityList):
        dictActivities[activity] = i
    valueList = sorted(set(values))
    dictValues = {}
    for i, v in enumerate(valueList):
        dictValues[v] = i
    dictObs = {}
    count = 0
    for key in dictSensors.keys():
        if "M" or "AD" in key:
            dictObs[key + "OFF"] = count
            count += 1
            dictObs[key + "ON"] = count
            count += 1
        if "D" in key:
            dictObs[key + "CLOSE"] = count
            count += 1
            dictObs[key + "OPEN"] = count
            count += 1
        if "T" in key:
            for temp in range(0, int((max(temperature) - min(temperature)) * 2) + 1):
                dictObs[key + str(float(temp / 2.0) + min(temperature))] = count + temp

    XX = []
    YY = []
    X = []
    Y = []
    for kk, s in enumerate(sensors):
        if "T" in s:
            XX.append(dictObs[s + str(round(float(values[kk]), 1))])
        else:
            XX.append(dictObs[s + str(values[kk])])
        YY.append(dictActivities[activities[kk]])

    # for kk, s in enumerate(sensors):
    #     try:
    #         if "T" in s:
    #             XX.append(dictObs[s + str(round(float(values[kk]), 1))])
    #         else:
    #             XX.append(dictObs[s + str(values[kk])])
    #         YY.append(dictActivities[activities[kk]])
    #     except KeyError as e:
    #         print(f"Error encountered for sensor {s} with value {values[kk]} on line {kk + 1}: {e}")
    #         continue

    x = []
    for i, y in enumerate(YY):
        if i == 0:
            Y.append(y)
            x = [XX[i]]
        if i > 0:
            if y == YY[i - 1]:
                x.append(XX[i])
            else:
                Y.append(y)
                X.append(x)
                x = [XX[i]]
        if i == len(YY) - 1:
            if y != YY[i - 1]:
                Y.append(y)
            X.append(x)
    return X, Y, dictActivities


def convertActivities(X, Y, dictActivities, uniActivities, cookActivities):
    Yf = Y.copy()
    Xf = X.copy()
    activities = {}
    for i, y in enumerate(Y):
        convertact = [key for key, value in dictActivities.items() if value == y][0]
        activity = uniActivities[convertact]
        Yf[i] = int(cookActivities[activity] - offset)
        activities[activity] = Yf[i]

    return Xf, Yf, activities


if __name__ == '__main__':
    for filename in datasets:
        datasetName = filename.split("/")[-1]
        print('Loading ' + datasetName + ' dataset ...')
        X, Y, dictActivities = load_dataset(filename)

        X, Y, dictActivities = convertActivities(X, Y,
                                                 dictActivities,
                                                 mappingActivities[datasetName],
                                                 cookActivities[datasetName])

        print(sorted(dictActivities, key=dictActivities.get))
        print("n° instances post-filtering:\t" + str(len(X)))

        print(Counter(Y))

        X = np.array(X, dtype=object)
        Y = np.array(Y, dtype=object)

        X = pad_sequences(X, maxlen=max_lenght, dtype='int32')
        # print("Padded X shape:", X.shape)
        # print("First 5 samples of padded X:", X[:5])
        if not os.path.exists('npy'):
            os.makedirs('npy')

        np.save('./npy/' + datasetName + '-x.npy', X)
        np.save('./npy/' + datasetName + '-y.npy', Y)
        np.save('./npy/' + datasetName + '-labels.npy', dictActivities)


def getData(datasetName):
    X = np.load('./npy/' + datasetName + '-x.npy', allow_pickle=True)
    Y = np.load('./npy/' + datasetName + '-y.npy', allow_pickle=True)
    dictActivities = np.load('./npy/' + datasetName + '-labels.npy', allow_pickle=True).item()
    # print("X shape:", X.shape)
    # print("Y shape:", Y.shape)
    # print("First 5 samples of X:", X[:5])
    # print("First 5 samples of Y:", Y[:5])
    return X, Y, dictActivities