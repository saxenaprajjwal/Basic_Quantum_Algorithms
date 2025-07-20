
import numpy as np
from functools import cache
import matplotlib.pyplot as plt


def bitshift(x: int, shift: int, n_bits: int) -> int:
    '''
    Does right bitshifting in two's complement.
    Note: The leftmost bit is copied leftwards, i.e., 1010>>1 == 1101.

    x      (int): Value to be bitshifted.
    shift  (int): How far the value is shifted right, shift is non-negative.
    n_bits (int): Number of bits representing x.

    return (int): Right bitshifted x.
    '''
    x = x%(2**n_bits)
    leftBit = (x&(1<<(n_bits-1)))>>(n_bits-1)

    if shift > n_bits:
        return 0
        
    x  = x>>shift
    x += (((1<<shift)-leftBit)<<(n_bits-shift))%(2**n_bits)

    return x


def add(
        x: int, y: int, shift: int, n: int = 8, negativeY: bool = False
    ) -> tuple[int, int]:
    '''
    Bitshifts y in two's complement and adds result to x.
    Should be unitary.
    Note: when bitshifting right, copies most significant bit left.
    
    x          (int): Number to be added to
    y          (int): Number adding into x
    shift      (int): Bitshift applied to y before addition
    n          (int): Number of bits
    negativeY (bool): Subtracts y if true
    '''
    N = 2**n
    x%=N; y%=N;

    if negativeY:
        x -= bitshift(y, shift, n)
    else:
        x += bitshift(y, shift, n)

    return x%N, y%N

@cache
def fib(i:int) -> int: 
    '''
    Fibonacci numbers { 1,1,3,5,8,13, ... }

    i (int): Index of Fibonacci number

    return (int): i^th Fibonacci number
    '''
    return int(np.round(
        1/(5**.5) * (((1+5**.5)/2)**(i+1)-((1-5**.5)/2)**(i+1))
    ))


def mult(
        x: int, aux: int, n: int, m: int, _debug: bool = False
    ) -> tuple[int, int]:
    '''
    In place multiplication integer by 1+2^(-m) with some error depending on 
    how clean the auxiliary registers are.
    Note: the aux register can gain some error

    x     (int): Value being multiplied
    aux   (int): auxiliary register
    n     (int): Number of bits representing x
    m     (int): x is multiplied by 1+2^(-m)

    return (tuple[int, int, int]):
        newX (int): the new value for x
        newAux (int): New value for aux, should equal to init value
    '''

    numIter = 2*int(.5*np.sqrt(5)*n/((1+np.sqrt(5))/2)**(m))
    if m>=n:
        return x, aux

    if _debug: print(f"in:\t[{x}, {aux}]")
    aux, x  = add(aux, x, 0, n=n)
    if _debug: print(f"init_aux:\t[{x}, {aux}]")


    for i in range(numIter, -1, -1):
        if m*fib(i) >= n: continue
        if i%2 == 0:
            x, aux = add(x, aux, m*fib(i), n, negativeY=not (fib(i)%2==1))
            if _debug: print(f"bounce(aux->x):\t[{x}, {aux}]")
        else:
            aux, x = add(aux, x, m*fib(i), n, negativeY=not (fib(i)%2==1))
            if _debug: print(f"bounce(x->aux):\t[{x}, {aux}]")

    aux, x = add(aux, x, 0, n, negativeY=True)
    if _debug: print(f"clear_aux:\t[{x}, {aux}]")

    return x, aux



def isneg(value: int, n_bits: int) -> bool:
    '''
    Returns true if the value is negative in two's complement

    value  (int): The value being assessed
    n_bits (int): Number of bits    

    return (bool): True if value is negative
    '''
    return (value & (1<<(n_bits-1))) != 0


def qasinModuloCORDIC(t: int, n_bits: int, _debug: bool = False) -> float:
    '''
    Unitary version of double rotation CORDIC algorithm

    Main source (for classical implementation):
        Computing Functions cos^-1 and sin^-1 Using Cordic
        IEEE Transactions on Computers, VOL. 42, NO. 1, January 1993
        Christophe Mazenc, Xavier Merrheim, and Jean-Michel Muller
    Note: the paper's algorithm seemed to have a bug, but I fixed it for this.

    Returns theta_i which approximates arcsin(t/(2^n_bits))

    t (int) [-(1<<n_bits),1<<n_bits]: Input angle written in 
        fixed point notation two's complement
    n_bits (int): Number of bits used to describe t

    return (float): theta_{n_bits+2}
    '''
    n:          int = n_bits+2
    theta_i:    int = 0
    x_i:        int = (1<<(n_bits))
    y_i:        int = 0
    t_i:        int = t
    aux:        int = 0
    d:   list[bool] = []

    if _debug:
        print(f"in    :\t{[t_i, x_i, y_i, aux, d]}") #temp
        print(f"init_x:\t{[t_i, x_i, y_i, aux, d]}") #temp

    for i in range(1, n):
        #Note: The original paper's equation for d introduced some
        #      errors. This new equation seems to have fixed the issue
        a, b, c = isneg(x_i, n), isneg(y_i, n), isneg(t_i-y_i, n)
        d.append(
            # isneg(x_i, n) != (isneg(t_i - (0 if isneg(x_i, n) else y_i), n))

            # not (isneg(x_i, n) and isneg(y_i, n)) 
            # and (isneg(x_i, n) or isneg(t_i-y_i, n))
            (((a and b) != (a and c)) != a) != c
        )
        
        if _debug: print(f"calc_d{i}:\t{[t_i, x_i, y_i, aux, d]}") #temp

        if d[-1]:
            y_i = ~y_i % (1<<n)
            # x_i, y_i = y_i, x_i
        if _debug: print(f"ref_y {i}:\t{[t_i, x_i, y_i, aux, d]}") #temp
        for _ in range(2):
            x_i, y_i = add(x_i, y_i, i, n=n, negativeY=True)
            if _debug: print(f"rot_A{i}:\t{[t_i, x_i, y_i, aux, d]}") #temp
            y_i, aux = mult(
                y_i, aux, n, 2*i
            )
            if _debug: print(f"rot_B{i}:\t{[t_i, x_i, y_i, aux, d]}") #temp
            y_i, x_i = add(y_i, x_i, i, n=n, negativeY=False)
            if _debug: print(f"rot_C{i}:\t{[t_i, x_i, y_i, aux, d]}") #temp
        if d[-1]:
            y_i = ~y_i % (1<<n)
            # x_i, y_i = y_i, x_i

        if _debug: print(f"rot_xy{i}:\t{[t_i, x_i, y_i, aux, d]}") #temp

        # theta_i += int(2**n * 2*(-1 if d[-1] else 1)*np.arctan(2**(-i)))
        theta_i += 2*(-1 if d[-1] else 1)*np.arctan(2**(-i))
        t_i, aux = mult(
            t_i, aux, n, 2*i
        )

        if _debug: print(f"updt_t{i}:\t{[t_i, x_i, y_i, aux, d]}") #temp

    if _debug: print(f"out:\t{[t_i, x_i, y_i, aux, d]}") #temp

    #temp
    if _debug: print(f"{t=}\td={[1 if val else 0 for val in d]}")

    return theta_i
    # return int(theta_i)/2**n 


def main():
    n = 16
    n_bits    = n-2 #sorry for the dumb naming

    test      = np.linspace(-(1<<(n_bits)), (1<<(n_bits))-1, num=(1<<(n_bits+1)),
        dtype=np.int32)
    # expected  = np.round(2**n_bits*np.arcsin(test/(2**n_bits)))/(2**n_bits)
    expected  = np.arcsin(test/(2**n_bits))
    predicted = np.array([qasinModuloCORDIC(t, n_bits) for t in test])


    print(f"{np.max(np.abs(expected-predicted))          = :.6f}")
    print(f"{test[np.argmax(np.abs(expected-predicted))] = }")
    print(f"{np.mean(np.abs(expected-predicted))         = :.6f}")
    print(f"{np.median(np.abs(expected-predicted))       = :.6f}")

    if n <= 5:
        plt.scatter(test, expected,  label="Expected")
        plt.scatter(test, predicted, label="Predicted")
    else:
        plt.plot(test, expected,  label="Expected")
        plt.plot(test, predicted, label="Predicted")
    plt.title("CORDIC Approx")
    plt.legend()
    # plt.grid()
    # plt.xticks(test)
    # plt.yticks(outRange/(2**(n_bits+1)))
    # plt.grid()
    plt.show()

    print(qasinModuloCORDIC(12, n_bits, True))
    # print(mult(4, 0, n_bits+2, 2, True))

    


if __name__ == "__main__":
    main()

