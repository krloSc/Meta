def map(value,xmin,xmax,ymin,ymax):
    result = (value-xmin)*((ymax-ymin)/(xmax-xmin))+ymin
    return result


if __name__ == "__main__":
    print(map(-72.6783,-74,-59,0,6000))
