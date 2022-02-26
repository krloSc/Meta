def bi_interpolation(x,y,x1,y1,x2,y2,q11,q12,q21,q22):
    R1 = ((x2 - x)/(x2 - x1))*q11 + ((x - x1)/(x2 - x1))*q21
    R2 = ((x2 - x)/(x2 - x1))*q12 + ((x - x1)/(x2 - x1))*q22
    P = ((y2 - y)/(y2 - y1))*R1 + ((y - y1)/(y2 - y1))*R2
    print(P)

bi_interpolation(50,0,0,-100,100,0,-50,0,0,50)
