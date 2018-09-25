#!/usr/bin/python

import numpy

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    # cleaned_data = []
    # errors = numpy.abs(net_worths-predictions)
    # outliers = numpy.array(errors <= numpy.percentile(errors,90))
    # cleaned_data = numpy.array([ages[outliers], net_worths[outliers], errors[outliers]]).transpose()
    # return cleaned_data
    cleaned_data = []
    for i in range(0, len(predictions)):
        cleaned_data.append((ages[i], net_worths[i], abs(predictions[i]-net_worths[i])))
        cleaned_data = sorted(cleaned_data, key=lambda student: student[2])
    return cleaned_data[:78]
# net_worths = numpy.array([1,3,3,5,4,6])
# ages = numpy.array([1,2,3,4,5,6])
# predictions = numpy.array([1,2,3,4,5,10])
#
# outlierCleaner(predictions,ages,net_worths)
