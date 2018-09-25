#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle
import math

enron_data = pickle.load(open("./final_project/final_project_dataset.pkl", "rb"))

keys = list(enron_data.keys())
items = list(enron_data.items())

for key in keys:
    print("Key: ", key, " POI?: ", enron_data[key]["poi"], "Email: ", enron_data[key]['email_address'])
print('Sample Item: ', enron_data['LAY KENNETH L'])
print('Num Keys: ', len(keys))
print('Num Features', len(items[0][1]))
num_poi = sum(1 if enron_data[key]["poi"] else 0 for key in keys)
print('Num POI: ',num_poi)
print('Num Salaries: ',sum(0 if math.isnan(float(enron_data[key]["salary"])) else 1 for key in keys))
print('Num Emails: ',sum(0 if enron_data[key]["email_address"] == 'NaN' else 1 for key in keys))
no_payment = sum(1 if enron_data[key]["total_payments"] == 'NaN' else 0 for key in keys)
print('Num No Total Payments: ',no_payment)
print('% No Total Payments: ',no_payment/len(keys))
poi_no_payment = sum(1 if (enron_data[key]["total_payments"] == 'NaN' and enron_data[key]["poi"]) else 0 for key in keys)
print('% POI no total payments: ', poi_no_payment/num_poi)
print(enron_data['GRAMM WENDY L'])


print('Lay (Founder, Chairman, ex-CEO): ', enron_data['LAY KENNETH L']['total_payments'])
print('Skilling (CEO): ', enron_data['SKILLING JEFFREY K']['total_payments'])
print('Fastow (CFO): ', enron_data['FASTOW ANDREW S']['total_payments'])
