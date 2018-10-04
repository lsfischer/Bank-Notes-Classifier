#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 15:51:38 2018

@author: Lucas Fischer
@author: Joana Martins
"""
from assignment import Assignment   #Import Assignment class from assignment.py

assignment = Assignment("TP1-data.csv", ",")
print(assignment.logistic_reg())
