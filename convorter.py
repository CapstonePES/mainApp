def CO_convertor(CO):
    if isinstance(CO, str):
        return 0
    elif CO <= 1:
        return CO * 50 / 1
    elif 1 < CO <= 2:
        return 50 + (CO - 1) * 50 / 1
    elif 2 < CO <= 10:
        return 100 + (CO - 2) * 100 / 8
    elif 10 < CO <= 17:
        return 200 + (CO - 10) * (100 / 7)
    elif 17 < CO <= 34:
        return 300 + (CO - 17) * (100 / 17)
    elif CO > 34:
        return 400 + (CO - 34) * (100 / 17)
    else:
        return None

def pm25_convertor(pm25):
    if isinstance(pm25, str):
        return 0
    elif pm25 <= 30:
        return pm25 * 50 / 30
    elif 30 < pm25 <= 60:
        return 50 + (pm25 - 30) * 50 / 30
    elif 60 < pm25 <= 90:
        return 100 + (pm25 - 60) * 100 / 30
    elif 90 < pm25 <= 120:
        return 200 + (pm25 - 90) * (100 / 30)
    elif 120 < pm25 <= 250:
        return 300 + (pm25 - 120) * (100 / 130)
    elif pm25 > 250:
        return 400 + (pm25 - 250) * (100 / 130)
    else:
        return None

