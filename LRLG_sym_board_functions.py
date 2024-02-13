import numpy as np
import copy 

# Just because I got so tired of writing it
def print_board(board):
    N = len(board)
    
    for i in range(N):
        print(board[i])

# 'fill' denotes entry to use for filling. "X" for bc board, "O" for wc board
def pos_to_board(pos, n, fill):
    board = [["."]*(2*n) for i in range(2*n)]
    
    for i in range(0,len(pos[:,0])):
        board[pos[i,0]][pos[i,1]] = fill
        
    return board

def add_boards(bcboard, wcboard):
    N = len(bcboard)
    
    sum_boards = [["."] * N for i in range(N)]
    
    for i in range(0,N):
        for j in range(0,N):
            a = bcboard[i][j]
            b = wcboard[i][j]
            if (a == "."):
                sum_boards[i][j] = b
            else:
                if(b == "."):
                    sum_boards[i][j] = a
                else:
                    sum_boards[i][j] = "XO"
    return sum_boards

# Adjusting index from n to -n to be 0 to 2n-1
def board_ind(k, n):
    shift_k = k
    
    if (k < 0):
        shift_k = n - 1 - k
    elif (k > 0):
        shift_k = n - k
    else: 
        raise Exception(k, " not a valid [-n,..,-1,1,..,n] index.")
    return shift_k

# Inverse of above
def inv_board_ind(k, n):
    shift_k = k
    
    if (0 <= k < n):
        shift_k = n - k
    elif (k >= n):
        shift_k = -k + n - 1
    else: 
        raise Exception(k, " not a valid [0,..,2n-1] index.")
    return shift_k

# Adjusting index from -n to n to be 0 to 2n-1
def perm_ind(k, n):
    shift_k = k
    
    if (k < 0):
        shift_k = n + k
    elif (k > 0):
        shift_k = (n-1) + k
    else: 
        raise Exception("Not a valid [-n,..,-1,1,..,n] index.")
    return shift_k

# inverse of above
def inv_perm_ind(k, n):
    shift_k = k
    
    if (k < n):
        shift_k = k - n
    elif (k >= n) and (k < 2*n):
        shift_k = k-n-1
    else: 
        raise Exception("Not a valid [-n,..,-1,1,..,n] index.")
    return shift_k

# getting the symmetric index for a given index - was too stupid to do this more than once
def symm_ind(i,n):
    return (2*n - 1) - i

# getting sub-boards for recursive checkerboard games - turning a 2n x 2n board to a 2k x 2k board
def get_sub_board(checker_pos, k, n):
    new_pos = []
    for checker in checker_pos:
        if (n-k <= checker[0] and checker[0] <= (2*n - 1) - (n-k)) and (n-k <= checker[1] and checker[1] <= (2*n - 1) - (n-k)):
            new_checker = [checker[0] - (n-k), checker[1] - (n-k)]
            new_pos.append(new_checker)
            
    return new_pos
# lwf given by long word factors from above function
# returning moving_rc gives desc checker, asc checker, then symm desc, symm asc
# if moving across horizontal middle, only gives desc checker, asc checker
def get_bc_perm_set(n):
    w0 = [0] * (2*n)
    
    for i in range(0, n): 
        w0[i] = n-i
        w0[(2*n)-(i+1)] = -(n-i)
    
    perms = [w0]
    asc_pairs = []
    desc_pairs = []
    
    for k in range(0, n**2):
        bprev = perms[k]
        
        bnext = copy.deepcopy(bprev)
        seeking_i = True
        i = -n
        
        asc = []
        desc = []
        case = ""
        
        while seeking_i and i <= 0:
            
            seeking_j = True
            j = 0
            
            if(i<0):
                
                ind = bprev.index(i)
                
                while seeking_j and j < ind:
               
                    if (i == bprev[j] - 1):
            
                        bnext[ind] = bprev[ind] + 1
                        bnext[symm_ind(ind,n)] = -(bprev[ind] + 1)
                    
                        bnext[j] = bprev[j] - 1
                        bnext[symm_ind(j,n)] = -(bprev[j] - 1)
                    
                        # Recording moving rows and moving columns
                        # Notice increasing the permutation image by 1 corresponds to increasing the
                        # row of that black checker by our convention
                        
                        asc.append([perm_ind(bprev[j], n), j])               
                        desc.append([perm_ind(bprev[ind], n), ind])
                        
                        asc_symm = [symm_ind(desc[0][0], n), symm_ind(desc[0][1], n)]
                        desc_symm = [symm_ind(asc[0][0], n), symm_ind(asc[0][1], n)]
            
                        asc.append(asc_symm)
                        desc.append(desc_symm)
                    
                        seeking_j = False
                        seeking_i = False
                        
                        case = " case 1 "
          
                    else:
                        j += 1
            
            # Case where i = 0 represents gap between -1 and 1? I think
            # CROSSOVER OF MIDDLE HORIZONTAL
            elif(i == 0):
                r = 2*n -1 
                
                # find the latest negative number (should be where we swap next)
                while bprev[r]>0:
                    r -= 1
                    
                bnext[r] = bprev[symm_ind(r,n)]
                bnext[symm_ind(r,n)] = bprev[r]
                
                # saving row r and its column c as moving rc
                asc.append([perm_ind(bprev[symm_ind(r,n)], n), symm_ind(r,n)])
                desc.append([perm_ind(bprev[r], n), r])
                
                asc_symm = [symm_ind(desc[0][0], n), symm_ind(desc[0][1], n)]
                desc_symm = [symm_ind(asc[0][0], n), symm_ind(asc[0][1], n)]
            
                asc.append(asc_symm)
                desc.append(desc_symm)
                
                seeking_j = False
                seeking_i = False
                
                case = " case 2"
            
            i += 1
            
        asc_pairs.append(asc)
        desc_pairs.append(desc)
                    
        perms.append(bnext)

        
    return perms, desc_pairs, asc_pairs

# Note only pos_y has structure: decreasing from n-1 to 0
#     as reading columns backwards to place checkers
def perm_to_positions(perm, n):
    pos_x = []
    pos_y = []
    
    for i in range(0, len(perm)):
        j = board_ind(perm[i], n)
        pos_x.append(j)
        
        # done since perms are given in -n to n, but board x indices go n to -n
        pos_y.append(symm_ind(i,n))
        
    return pos_x, pos_y
    
def get_ngame_bc_pos(n):
    bcmoves, desc_pairs, asc_pairs = get_bc_perm_set(n)
    
    bc_pos = np.zeros((n**2+1, 2*n, 2), dtype = int)
    
    
    for i in range(0,len(bcmoves)):
        x_i_set, y_i_set = perm_to_positions(bcmoves[i], n)
        
        bc_pos[i,:,0] = x_i_set
        bc_pos[i,:,1] = y_i_set
    
    return bc_pos, desc_pairs, asc_pairs