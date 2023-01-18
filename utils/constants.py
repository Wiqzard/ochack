#CLASSES = {
    #"__background__": 0,
    #"stillage_close": 1,
    #"stillage_open": 2,
    #"l_klt_6147": 3,
    #"l_klt_8210": 4,
    #"l_klt_4147": 5,
    #"pallet": 6,
    #"jack": 7,
    #"forklift": 8,
    #"str": 9,
    #"bicycle": 10,
    #"dolly": 11,
    #"exit_sign": 12,
    #"fire_extinguisher": 13,
    #"spring_post": 14,
    #"locker": 15,
    #"cabinet": 16,
    #"cardboard_box": 17,
#}
#CLASSES_RE = {
    #0: "__background__",
    #1: "stillage_close",
    #2: "stillage_open",
    #3: "l_klt_6147",
    #4: "l_klt_8210",
    #5: "l_klt_4147",
    #6: "pallet",
    #7: "jack",
    #8: "forklift",
    #9: "str",
    #10: "bicycle",
    #11: "dolly",
    #12: "exit_sign",
    #13: "fire_extinguisher",
    #14: "spring_post",
    #15: "locker",
    #16: "cabinet",
    #17: "cardboard_box",
#}
#CLASSES_ID = {
    #1002: 0,
    #1003: 1,
    #1012: 2,
    #1013: 3,
    #1011: 4,
    #1100: 5,
    #1120: 6,
    #2010: 7,
    #2050: 8,
    #2000: 9,
    #1110: 10,
    #4000: 11,
    #5010: 12,
    #1135: 13,
    #1030: 14,
    #1040: 15,
    #1070: 16,
#}
#CLASSES_ID_RE = {value: key for key, value in CLASSES_ID.items()}
##CLASSES_ID = {
#    1002: 1,
#    1003: 2,
#    1012: 3,
#    1013: 4,
#    1011: 5,
#    1100: 6,
#    1120: 7,
#    2010: 8,
#    2050: 9,
#    2000: 10,
#    1110: 11,
#    4000: 12,
#    5010: 13,
#    1135: 14,
#    1030: 15,
#    1040: 16,
#    1070: 17,
#}
#CLASSES_ID_RE = {value: key for key, value in CLASSES_ID.items()}
#CLASSES_RE = {
#    1: "stillage_close",
#    2: "stillage_open",
#    3: "l_klt_6147",
#    4: "l_klt_8210",
#    5: "l_klt_4147",
#    6: "pallet",
#    7: "jack",
#    8: "forklift",
#    9: "str",
#    10: "bicycle",
#    11: "dolly",
#    12: "exit_sign",
#    13: "fire_extinguisher",
#    14: "spring_post",
#    15: "locker",
#    16: "cabinet",
#    17: "cardboard_box",
#}
#models = {1: [1, 2, 3, 4, 5], 2: [6, 15, 16, 17, 9, 8], 3: [7, 10, 11, 12, 13, 14]}
#model_classes = {
#    1: {
#        0: "l_klt_4147",
#        1: "l_klt_6147",
#        2: "l_klt_8210",
#        3: "stillage_close",
#        4: "stillage_open",
#    },
#    2: {
#        0: "pallet",
#        1: "cabinet",
#        2: "cardboard_box",
#        3: "locker",
#        4: "forklift",
#        5: "str",
#    },
#    3: {
#        0: "bicycle",
#        1: "dolly",
#        2: "exit_sign",
#        3: "fire_extinguisher",
#        4: "jack",
#        5: "spring_post",
#    },
#}
#
# model_classes = {
#    1: {
#        3: "stillage_close",
#        4: "stillage_open",
#        1: "l_klt_6147",
#        2: "l_klt_8210",
#        0: "l_klt_4147",
#    }[ 'l_klt_4147', 'l_klt_6147', 'l_klt_8210', 'stillage_close', 'stillage_open'],
#    2: {
#        0: "pallet",
#        3: "locker",
#        1: "cabinet",
#        2: "cardboard_box",
#        5: "str",
#        4: "forklift",
#    }[ 'pallet', 'cabinet', 'cardboard_box', 'locker', 'forklift','str'],
#    3: {
#        4: "jack",
#        0: "bicycle",
#        1: "dolly",
#        2: "exit_sign",
#        3: "fire_extinguisher",
#        5: "spring_post",
#    }['bicycle', 'dolly', 'exit_sign', 'fire_extinguisher', 'jack', 'spring_post'],
# }
models = {1: [1, 2, 3, 4, 5]}

model_classes = {1: {
    0 : "klt_box_empty",
    1 : "klt_box_full",
    2 : "rack_1",
    3 : "rack_2",
    4 : "rack_3",
}}