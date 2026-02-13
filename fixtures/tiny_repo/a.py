import helpers
from classes import Accumulator
from helpers import helper_sum


def add_numbers(nums):
    total = 0
    for n in nums:
        total += n
    return total


def sum_list(values):
    result = 0
    for value in values:
        result += value
    return result


def wrapper(values):
    return add_numbers(values)


def via_import(values):
    return helper_sum(values)


def via_module(values):
    return helpers.helper_sum(values)


def via_instance(values):
    acc = Accumulator()
    return acc.total(values)
