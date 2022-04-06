import json


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def getmax():
    likelihood = json.load(open("likelihood_images.json"))

    max_like = {}
    for key in likelihood.keys():
        value = likelihood[key]
        name = key.split('_')[0]
        for i in value:
            if is_number(i):
                continue
            else:
                max_name = i.split('_')[0]
                if max_name == name:
                    continue
                else:
                    max_like[key.split('.')[0]+'.jpg'] = i.split('.')[0]+'.jpg'
                    break

    f = open("maxlikelihood_images.json", "w")
    f.write(json.dumps(max_like))
    f.close()

if __name__ == '__main__':
    getmax()

