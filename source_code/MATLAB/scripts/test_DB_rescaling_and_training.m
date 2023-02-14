close all
clear all

addpath('../functions');

resol=[28,28];
limit_output=1;
database='MNIST';

DB_rescale_ANN_train(resol,limit_output,database);