#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:shy
@license: Apache Licence 
@file: MTLShow.py 
@time: 2020/09/26
@contact: justbeshy@outlook.com
@site:  
@software: PyCharm 

@description:

# Programs must be written for people to read.
# Good code is its own best documentation.
# Focus on your question, not your function.
"""
from torchvision import transforms

def showTensor(output):
    save_img = transforms.ToPILImage()(output).convert('L')
    # path = '../Log/' + args.logdir + '/V' + str(j) + '/'
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # save_img.save(path + 'E' + str(epoch) + '_' + ID[j] + '.jpg')