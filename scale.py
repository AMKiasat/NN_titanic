def scale_into_number(v):
    s = [int(v[2])]

    if v[4] == 'male':
        s.append(1)
    elif v[4] == 'female':
        s.append(2)
    else:
        s.append(v[4])

    if v[5] == '':
        s.append(0)
    else:
        s.append(int(float(v[5])))

    s.append(int(v[6]))
    s.append(int(v[7]))
    s.append(int(float(v[9])))

    if v[11] == 'S':
        s.append(1)
    elif v[11] == 'C':
        s.append(2)
    elif v[11] == 'Q':
        s.append(3)
    else:
        s.append(-1)

    s.append(int(v[1]))

    return s
