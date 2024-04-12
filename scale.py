def scale_into_number(v):
    s = [int(v[2])]

    if v[4] == 'male':
        s.append(0)
    elif v[4] == 'female':
        s.append(1)
    else:
        s.append(v[4])

    if v[5] == '':
        s.append(0)
    else:
        s.append(float(v[5]))

    s.append(int(v[6]))
    s.append(int(v[7]))
    s.append(float(v[9]))

    if v[11] == 'S':
        s.append(0)
    elif v[11] == 'C':
        s.append(1)
    elif v[11] == 'Q':
        s.append(2)
    else:
        s.append(v[11])

    s.append(v[1])

    return s
