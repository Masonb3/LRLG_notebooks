import numpy as np
import copy

from LRLG_sym_board_functions import *

# LATER: Replace len(wc_x) with n? I think there always have to be n wcs on any board


# lam, mu each arrays of strictly decreasing integers between 1, n
#
# RETURNS: pos_x, pos_y of white checkers on the board
def init_white_config(lam, mu, n):
    
    A = lam
    B = mu
    
    # HAD TO ADJUST INDEXING HERE
    for i in range(1,n+1):
        if not (i in lam):
            A.append(-i)
        if not (i in mu):
            B.append(-i)
    
    #print("A: ", A)
    #print("B: ", B)
    
    wc_pos = np.zeros((n,2), dtype=int)
    
    for i in range(0, n):
        wc_pos[i,0] = board_ind(B[(n-1) - i],n)
        wc_pos[i,1] = board_ind(A[i],n)
        
    return  wc_pos

# Note here that wc_x and wc_y define respective positions for ALL white checkers
def get_wc_degree(wc_index, bc_pos, wc_pos, n):
    
    wc_x = wc_pos[:,0]
    wc_y = wc_pos[:,1]
    
    bc_x = bc_pos[:,0]
    bc_y = bc_pos[:,1]
    
    x = wc_x[wc_index]
    y = wc_y[wc_index]
    
    L = wc_index
    
    bc_NW = 0
    wc_NW = 0
    sigma = 0
    
    for i in range(0,len(bc_x)):
        if (bc_x[i] <= x and bc_y[i] <= y):
            bc_NW += 1
        
        
    for i in range(0,len(wc_x)):
        # Note by this convention we include wc_pos itself in wc_NW,
        #     so it will not be subtracted separately later
        if (wc_x[i] <= x and wc_y[i] <= y):
            wc_NW += 1
    
    # This for loop assumes wc_x and wc_y are ordered such that
    # wc_y is in increasing order (Left to Right reading on columns)
    for j in range(0,L):
        eps = 0
        for i in range(len(bc_x)):
            bpair_x = symm_ind(bc_x[i], n)
            bpair_y = symm_ind(bc_y[i], n)
            
            if (bc_x[i] <= x and bc_y[i] <= y) and (bpair_x <= wc_x[j] and bpair_y <= wc_y[j]):
                eps = 1
              
        # only need to check one case since we iterate over all bc, so 
        # loop will bc_x,bc_y as both a bc and later as its pair
        sigma += eps
    
    wc_deg = bc_NW - wc_NW - sigma
    
    #print(" BC - WC - PAIRS ", bc_NW, wc_NW, sigma)
    
    if (wc_deg < 0): 
        wc_deg = 0 
    
    return wc_deg

def get_total_wc_degree(bc_pos, wc_pos, n):

    total_deg = 0
    
    for i in range(0, len(wc_pos[:,0])):
        cur_deg = get_wc_degree(i, bc_pos, wc_pos, n)
        print(cur_deg)
        total_deg += cur_deg
        
    return total_deg

def test_wc_methods(lam, mu, n, wc_index, bcboard_index):

    ngame_pos, asc, desc = get_ngame_bc_pos(n)
    wc_pos = init_white_config(lam,mu,n)
    
    bc_pos = ngame_pos[bcboard_index]
    
    B_board = pos_to_board(bc_pos, n, "X")
    W_board = pos_to_board(wc_pos, n, "O")

    T = add_boards(B_board, W_board)
    
    degs = []
    for i in range(0, len(wc_pos)):
        
        degs.append(get_wc_degree(i, bc_pos, wc_pos, n))
    
    tdeg = get_total_wc_degree(bc_pos, wc_pos, n)
    
    print_board(T)
    print("Degrees of wcs", degs)
    print("Total degree is ", tdeg)

# index specifies which white checker we are checking
# we assume bc_pos forms a valid board
def check_happy(bc_pos, wc_pos, wc_test, n):
    
    to_check_x = wc_test[0]
    to_check_y = wc_test[1]
    
    is_happy = True
    reasons = []
    
    # 2. Check there is a black checker in same row, col
    # to left or above resp.
    bc_row_ind = np.where(bc_pos[:,0] == to_check_x)[0][0]
    bc_col_ind = np.where(bc_pos[:,1] == to_check_y)[0][0]
    
    bc_row = bc_pos[bc_row_ind]
    bc_col = bc_pos[bc_col_ind]
    
    if (bc_row[1] > to_check_y):
        is_happy = False
        reasons.append(" Black checker in row " + str(to_check_x) + " has column " + str(bc_row[1]) + " too high.")
    if (bc_col[0] > to_check_x):
        is_happy = False
        reasons.append(" Black checker in column " + str(to_check_y) + " has row " + str(bc_col[0]) + " too high.")
        
    # 3. At most one wc in row, col
    # 4. No wc in symmetric row or column
    for i in range(0, len(wc_pos)):
        if (to_check_x != wc_pos[i][0] or to_check_y != wc_pos[i][1]): 
            if (wc_pos[i,0] == to_check_x): 
                is_happy = False
                reasons.append(" White checker in same row (row " + str(to_check_x) + "). ")
            elif (wc_pos[i,1] == to_check_y):
                is_happy = False
                reasons.append(" White checker in same col (col " + str(to_check_y) + "). ")
            elif (wc_pos[i,0] == symm_ind(to_check_x,n)):
                is_happy = False
                reasons.append(" White checker in symm row (row " + str(to_check_x) + ", symm " + str(symm_ind(to_check_x,n)) + "). ")
            elif (wc_pos[i,1] == symm_ind(to_check_y,n)):
                is_happy = False
                reasons.append(" White checker in symm col (col " + str(to_check_y) + ", symm " + str(symm_ind(to_check_y,n)) + "). ")
            
    return is_happy, reasons

# Takes bc_pos (of resulting board from bc move) and moves white checker left until happy

def get_move_left(bc_pos, wc_pos, wc_ind, n):
    
    wc_new_config = wc_new_config_x = copy.deepcopy(wc_pos)
    
    print("TRYING TO MOVE LEFT WITH WC POS: ", wc_pos)
    print("TRYING TO MOVE LEFT WITH WC_IND GIVEN AS ", wc_ind)
    
    if type(wc_ind) == int:
        wc0 = wc_pos[wc_ind]
        
        j = wc0[1] # column to be shifted 
        happiness = False
        
        while not happiness and j >= 0:
            wc_new_config[wc_ind] = [wc0[0], j-1]
            # Moving one to the left 
            curr_happiness, reasons = check_happy(bc_pos, wc_new_config, wc_pos[wc_ind], n) 
            if (curr_happiness): 
                happiness = True
            else:
                j -= 1
        if j == -1:
            raise Exception("Tried to move left wc at [" + str(wc0[0]) + ", " + str(wc0[1]) + "] but couldn't make happy.")
        
    elif len(wc_ind) == 2:
        # Moving left until attached to two black checkers
        wc0 = wc_pos[wc_ind[0]]
        wc1 = wc_pos[wc_ind[1]]
        
        a = np.where(bc_pos[:,0] == wc0[0])[0][0]
        b = np.where(bc_pos[:,0] == wc1[0])[0][0]
        
        bc0 = bc_pos[a] 
        bc1 = bc_pos[b]
        
        wc_new_config[wc_ind[0]] = bc0
        wc_new_config[wc_ind[1]] = bc1
        
    else:
        raise Exception("Can only perform a 'move left' with one or two checkers.")
        
    return wc_new_config

# Return -1 if no checker found in SW
def find_in_SW(wc_pos, n):
    wc_high_ind = -1
    for i in range(0,n):
        curr_row = wc_pos[i][0]
        curr_col = wc_pos[i][1]
        SW = (curr_row >= n) and (curr_col <= n-1)
        if(SW and wc_high_ind == -1):
            wc_high_ind = i
        elif(SW and curr_row < wc_pos[wc_high_ind][0]):
            wc_high_ind = i
            
    return wc_high_ind

def get_SE(wc_pos, n):
    wc_SE = []
    for i in range(0,n):
        curr_row = wc_pos[i][0]
        curr_col = wc_pos[i][1]
        if (curr_row >= n) and (curr_col >= n):
            wc_SE.append(wc_pos[i])
    return wc_SE

def get_SW(wc_pos, n):
    wc_SW = []
    for i in range(0,n):
        curr_row = wc_pos[i][0]
        curr_col = wc_pos[i][1]
        if (curr_row >= n) and (curr_col < n):
            wc_SW.append(wc_pos[i])
    return wc_SW

def get_high_SW(wc_pos, n):
    wc_SW = [-1,2*n]
    ind = 0
    for i in range(0,n):
        curr_row = wc_pos[i][0]
        curr_col = wc_pos[i][1]
        if (curr_row >= n) and (curr_col < n) and (curr_row < wc_SW[1]):
            wc_SW = wc_pos[i]
            ind = i
    return wc_SW, ind

def get_right_SE(wc_pos, n):
    wc_SE = [-1,-1]
    ind = 0
    for i in range(0,n):
        curr_row = wc_pos[i][0]
        curr_col = wc_pos[i][1]
        if (curr_row >= n) and (curr_col >= n) and (curr_col > wc_SE[1]):
            wc_SE = wc_pos[i]
            ind = i
    return wc_SE, ind

def exists_blocker(wc_pos, wc1, wc2, n):
    exists = False
    
    min_row = min(wc1[0],wc2[0])
    min_col = min(wc1[1],wc2[1])
    max_row = max(wc1[0],wc2[0])
    max_col = max(wc1[1],wc2[1])
    
    # 'rightmost such blocker'
    wc03 = [-1,-1]
    
    for i in range(0, n):
        curr_row = wc_pos[i][0]
        curr_col = wc_pos[i][1]
        
        # we do not count the given wc1, wc2 as blockers
        is_wc_given = (curr_row == wc1[0] and curr_col == wc1[1]) or (curr_row == wc2[0] and curr_col == wc2[1])
        
        if (not is_wc_given and 
           (min_row <= curr_row and curr_row <= max_row) and
           (min_col <= curr_col and curr_col <= max_col)):
            exists = True
            if curr_col > wc03[1]:
                wc03[0] = curr_row
                wc03[1] = curr_col
                
    return exists, wc03

# Takes board with white checker on either NW or SE diagonal and pops it, 
# returning partition value and clean board
def pop_board(wc_pos,n):
    clean_board = []
    
    part_val = 0
    skip_index = -1
    if(0 in wc_pos[:,0]):
        skip_index = 0
        part_val = n
    else: 
        skip_index = 2*n - 1 
        part_val = -n
        
    for i in range(0, len(wc_pos)):
        wc = wc_pos[i]
        if (wc[1] != skip_index):
            # Shift down since board is getting smaller
            clean_board.append([wc[0]-1, wc[1]-1])
        
    return np.array(clean_board, dtype=int), part_val

################################################################################################################## CASE MOVES

def get_case1_move(wc_pos, asc_ind):
    # TWO MUST ASCEND AT THE SAME TIME OR SYMMETRY CONDITION IS VIOLATED
    
    temp_board = copy.deepcopy(wc_pos)
    k = len(asc_ind)
    if (k != 2):
        raise Exception("Expect 2 checkers to ascend for get_case1_move, but ", k, " checkers want to move.")
    else:
        temp_board[asc_ind[0],0] -= 1
        temp_board[asc_ind[1],0] -= 1
    mult = 1
    
    return temp_board, mult
    
def get_case2_move(wc_pos, ind_a, ind_b):

    temp_board = copy.deepcopy(wc_pos)
    
    temp_col = temp_board[ind_a, 1]
    temp_board[ind_a, 1] = temp_board[ind_b, 1]
    temp_board[ind_b, 1] = temp_col
    
    mult = 1
    
    return temp_board, mult

def get_horiz_move(bc_pos, bc_pos_next, wc_pos, n):
    
    res_boards = []
    res_mults = []
    res_cases = []
    
    wc_x = wc_pos[:,0]
    wc_y = wc_pos[:,1]
    
    OLD_TOTAL_DEG = get_total_wc_degree(bc_pos, wc_pos, n)
    
    if (n == 1):
        # only one white checker, and if it is in bottom right, it stays, otherwise it moves to top left
        small_board = copy.deepcopy(wc_pos)
        if not (wc_pos[0, 0] == n and wc_pos[0, 1] == n):
            small_board[0, 1] = 0
            small_board[0, 0] = 0
        res_boards.append(small_board)
        res_mults.append(1)
        res_cases.append("n=1 move")
     
    else:
        # Handling white checker on row below horizontal midline
        if (n in wc_x):
            print("Horizontal move with wc on lower row, n = ", n)
            print ("WCPOS", wc_pos)
            temp_board = copy.deepcopy(wc_pos)
            
            wc_low_ind = np.where(wc_x == n)[0][0]
            if(temp_board[wc_low_ind, 1] == 2*n -1):
                # If checker is on last column, it can stay
                res_boards.append(temp_board)
                res_mults.append(1)
                res_cases.append("Horizontal cross with wc stay")
            else:
                # Otherwise, needs to move up
                temp_board[wc_low_ind, 0] -= 1
                res_boards.append(temp_board)
                res_mults.append(1)
                res_cases.append("Horizontal cross with wc ascend")
            
        # Case for checker above horizontal midline
        else:
            wc_onr_ind = np.where(wc_y == 2*n-1)[0][0]

            horiz_stay_config = copy.deepcopy(wc_pos)
            horiz_move_config = copy.deepcopy(wc_pos)

            moved_far = False
            stayed = False

            # FAR MOVE - WC CROSSES HORIZ
            move_search = True
            sw_to_check = get_SW(wc_pos, n)
            search_count = 0

            # Maybe we don't need to move a SW checker to compensate
            horiz_move_config[wc_onr_ind, 1] = 0
            test_move_deg = get_total_wc_degree(bc_pos_next, horiz_move_config, n)
            if test_move_deg == OLD_TOTAL_DEG:
                move_search = False
                moved_far = True

            while move_search and search_count < len(sw_to_check):
                sw_wc_curr = sw_to_check[search_count]
                sw_wc_curr_ind = np.where(wc_y == sw_wc_curr[1])[0][0]
                horiz_move_config[sw_wc_curr_ind, 1] = symm_ind(sw_wc_curr[1], n)
                horiz_move_config[wc_onr_ind, 1] = 0
                test_move_deg = get_total_wc_degree(bc_pos_next, horiz_move_config, n)
                if test_move_deg == OLD_TOTAL_DEG: 
                    move_search = False
                    moved_far = True
                else:
                    # Reset the config to test and move to the next SW checker
                    horiz_move_config = copy.deepcopy(wc_pos)
                    search_count += 1

            # STAY MOVE - swap with rightmost wc in SE region
            se_to_swap, se_to_swap_ind = get_right_SE(wc_pos, n)

            horiz_stay_config[se_to_swap_ind, 1] = 2*n-1
            horiz_stay_config[wc_onr_ind, 1] = se_to_swap[1]

            test_stay_deg = get_total_wc_degree(bc_pos_next, horiz_stay_config, n)
            if test_stay_deg == OLD_TOTAL_DEG: 
                stayed = True

            if moved_far:
                res_boards.append(horiz_move_config)
                res_mults.append(2)
                res_cases.append("Far move on horizontal cross")

            if stayed:
                res_boards.append(horiz_stay_config)
                res_mults.append(1)
                res_cases.append("Stay move on horizontal cross")

            if (not moved_far) and (not stayed):
                    raise Exception("W/C did not move properly at horizontal cross")            

    return res_boards, res_mults, res_cases

def get_case3_move(bc_pos, bc_pos_next, wc_pos, desc, asc, n, stage):
    
    res_boards = []
    res_mults = []
    res_cases = []
    
    wc_x = wc_pos[:,0]
    wc_y = wc_pos[:,1]
    
    if(stage == 1):
        # CASE 3: Awc = Attached Checker
        row_i = min(desc[0][0], desc[1][0])
        col_j = min(desc[0][1], desc[1][1])
        # MIGHT NEED TO FIX COL_J
        
        l = col_j - min(asc[0][1], asc[1][1]) 
        
        if (row_i not in wc_x) or wc_pos[np.where(wc_x == row_i)[0], 1] != bc_pos[np.where(bc_pos[:,0] == row_i)[0], 1]:
            # Shouldn't happen as given in cases
            raise Exception("In Stage 1 Case 3, but no attached white checker was found at i=" + str(row_i))
        else:
            # GET POSITIVE I from desc black checkers
            if row_i < n-1:
                opp_im = symm_ind(board_ind(inv_board_ind(row_i,n) - 1, n), n) 
                
                temp_config = copy.deepcopy(wc_pos)
                if (col_j in wc_y) and (wc_pos[np.where(wc_y == col_j)[0], 0] == opp_im):
                    # CASE 3.1  
                    
                    # First check if there is a white checker one to the right of the symm of the unhappy one
                    # use minus l because of array indexing (this is opposite convention of decreasing board indexing)
                    if (col_j - l + 1) in wc_y:
                        sad_NE_ind = np.where(wc_y == symm_ind(col_j, n) + l)[0][0]
                        sad_NE_wc = wc_pos[sad_NE_ind]
                        
                        jplm1 = symm_ind(sad_NE_wc[1], n) + 1
                        
                        swap_3_1_ind = np.where(wc_y == jplm1)[0][0]

                        if not (swap_3_1_ind >= 0):
                            raise Exception ("Stage 1 Case 3.1. Trying to move unhappy wc 1 to left, but cannot find wc on j+l-1 (board convention)")
                        else:
                            swap_3_1_wc = wc_pos[swap_3_1_ind]
                        
                        on_col_j_ind = np.where(wc_y == col_j)[0][0]
                        on_col_j_wc = wc_pos[on_col_j_ind]
                        
                        # unhappy moves 1 to left, forces swap to move to j, forces j to move to symm of previous unhappy col
                        temp_config[sad_NE_ind, 1] -= 1
                        temp_config[swap_3_1_ind, 1] = on_col_j_wc[1]
                        temp_config[on_col_j_ind, 1] = col_j - l
                        
                    else:
                        moved_left = get_move_left(bc_pos_next, wc_pos, [np.where(wc_x == row_i)[0][0], np.where(wc_y == col_j)[0][0]], n)
                        temp_config = moved_left
                    
                    res_boards.append(temp_config)
                    res_mults.append(1)
                    res_cases.append("Stage 1 Case 3.1")
                else:
                    # CASE 3.2
                    wc0_ind = np.where(wc_x == row_i)[0][0]
                    wc0 = wc_pos[wc0_ind]
                    
                    # CASE 3.2.1.
                    # no checks since must occur following failure of 3.1?
                    sw_i = wc0[0] 
                    sw_j = wc0[1]
                    
                    sw_ind = -1
                    
                    if (sw_i != 2*n-1 and sw_j != 0):
                        searching = True
                        
                        sw_i += 1
                        sw_j -= 1
                        
                        while searching:
                            # Assuming right-most WC takes priority SW of checker in question
                            if (sw_j in wc_y) and (wc_pos[np.where(wc_y == sw_j)[0],0] > sw_i):
                                searching = False
                                sw_ind = np.where(wc_y == sw_j)[0]
                                sw_j = wc_pos[sw_ind,1]
                                
                            elif (sw_j == 0):
                                raise Exception("No SW checker found in case 3.1.1, O checker is [" + str(wc0[0]) + ", " + wc0[1] +"].")
                            else:
                                sw_j -= 1 
                    else:
                        raise Exception("Stage 1 Case 3.2.1 tried to find a SW to swap with, but given wc was too far left/down.")
                        
                    # Found a sw checker for case 3.2.1 as expected
                    temp_config[sw_ind, 1] = wc0[1]
                    temp_config[wc0_ind, 1] = sw_j
                
                    res_boards.append(temp_config)
                    res_mults.append(1)
                    res_cases.append("Stage 1 Case 3.2.1")
                        
                    # CASE 3.2.2.
                        # Shift columns, checking to see if there is a checker that can be shifted
                        # after moving O to the right by 1
                    s0 = wc0[1] - 1
                    
                    if(symm_ind(s0, n) in wc_y):
                        
                        split_config = copy.deepcopy(wc_pos) 
                        
                        wc01_ind = np.where(wc_y == symm_ind(s0,n))[0][0]
                        wc01 = wc_pos[wc01_ind]
                        
                        # I believe wc02 must always be either on col 1 to right of wc01
                        # or on the symm of that col - NOT SURE IF THIS IS TRUE
                        
                        wc02_ind = -1
                        if (wc01[1] + 1 in wc_y):
                            wc02_ind = np.where(wc_y == wc01[1] + 1)[0][0]
                        else :
                            # in symm of 1 to right of wc01
                            wc02_ind = np.where(wc_y == symm_ind(wc01[1] + 1,n))[0][0]
                            
                        wc02 = wc_pos[wc02_ind]
                        
                        s1 = wc01[1]
                        s2 = wc02[1]
                        
                        split_config[wc02_ind, 1] = s1 - 1 
                        split_config[wc01_ind, 1] = s2
                        split_config[wc0_ind, 1] = s0
                        
                        m = 1
                        if (col_j + l) == n and (wc01[0] == symm_ind(row_i + 1, n)):
                            m = 2
                            
                        res_boards.append(split_config)
                        res_mults.append(m)
                        res_cases.append("Stage 1 Case 3.2.2")
        
    elif(stage == 2):
        temp_config = copy.deepcopy(wc_pos)
        
        row_i = min(desc[0][0], desc[1][0])
        
        moving_col = np.array([desc[0][1], desc[1][1], asc[0][1], asc[1][1]])
        moving_col = np.sort(moving_col)
        
        # idea here is that moving_col sorted will be of form {n, j, -j, -n}
        col_j = moving_col[1]
        #print("COL J", col_j)
        
        wc0_ind = np.where(wc_x == row_i)[0][0]
        wc01_ind = -1
        wc02_ind = -1
        
        wc0 = wc_pos[wc0_ind]
        wc01 = [-1,-1]
        wc02 = [-1,-1]
        
        if (col_j > 0):
            row_mim1 = symm_ind(row_i + 1, n)
            #print("ROW MIM1", row_mim1)
            wc01_ind = np.where(wc_x == row_mim1)[0][0]
            wc01 = wc_pos[wc01_ind]
        elif (n==1):
            wc01_ind = wc0_ind
            wc01 = wc0
        
        poss_02_ind = find_in_SW(wc_pos, n)
        if (poss_02_ind != -1):
            wc02_ind = poss_02_ind
            wc02 = wc_pos[wc02_ind]
            
        # All of case 3 assumes wc02 exists?
        blocked, wc03 = exists_blocker(wc_pos, wc0, wc02, n)
        SE_wc = get_SE(wc_pos, n)
        SW_wc = get_SW(wc_pos, n)
        
        # Second condition ensures wc02 exists - cannot be considered blocked if wc02 does not exist
        if blocked and wc02_ind >= 0:
            # Case 3.1
            wc03_ind = np.where(wc_x == wc03[0])[0][0]
                
            print("wc03 is at ", wc03)
            print("wc0 is at ", wc0)
            print("wc01 is at ", wc01)
            print("wc02 is at ", wc02)
            temp_config[wc0_ind, 1] = wc03[1]
            # NEW 3.1 ADDED AFTER FAILURE AT [2,1]*[2]
            temp_config[wc03_ind, 1] = wc01[1]
            temp_config[wc01_ind, 1] = wc0[1]
            
            res_boards.append(temp_config)
            res_mults.append(1)
            res_cases.append("Stage 2 Case 3.1 (blocked)")
                    
        # Second part of elif is wc02 not existing, since this means there can be no blocker
        elif (not blocked or wc02_ind == -1):
            # Split cases
            
            # Case 3.2.1
            temp_config[wc0_ind, 1] = 0
                
            testing_config = copy.deepcopy(temp_config)
                
            made_02_happy = False
            made_01_happy = False
            
            new_02_col = wc02[1] + 1
            new_01_col = wc01[1] - 1
            
            if (wc02_ind != -1):
                while not made_02_happy and new_02_col < 2*n:
                    wc02_test = [wc02[0], new_02_col]
                    # changing testing_config to make check_happy not test against wc02_test
                    testing_config[wc02_ind] = wc02_test
                    is_hap, r = check_happy(bc_pos_next, testing_config, wc02_test, n)
                    if is_hap:
                        made_02_happy = True
                        temp_config[wc02_ind, 1] = new_02_col
                    else:
                        new_02_col += 1
            else:
                made_02_happy = True
            
            while not made_01_happy and new_01_col >= 0:
                wc01_test = [wc01[0], new_01_col]
                # changing testing_config to make check happy not test against wc01_test
                testing_config[wc01_ind] = wc01_test
                is_hap, r = check_happy(bc_pos_next, testing_config, wc01_test, n)
                if is_hap:
                    made_01_happy = True
                    temp_config[wc01_ind, 1] = new_01_col
                    #print(" MADE 01 HAPPY from ", wc01[1], " to ", new_01_col)
                else:
                    new_01_col -= 1
                        
            if not made_02_happy:
                raise Exception("Case 3.2.1 could not move 02 right to be happy. wc0 at ", wc0, " wc01 at ", wc01, " wc02 at ", wc02)
            elif not made_01_happy:
                raise Exception("Case 3.2.1 could not move 01 left to be happy. 01 at ", wc01)
            else:
                res_boards.append(temp_config)
                res_mults.append(1)
                res_cases.append("Stage 2 Case 3.2.1")
                    
            # Case 3.2.2
                # first finding if any checkers in SE are SW of wc0
            wc04_ind = -1
            wc04 = [-1,-1]
        
            # Three possible swaps - move wc0 one to left, to -1 column (1 right of vertical midline), or to n column (far left)
        
            for i in range(0, len(SE_wc)):
                if (SE_wc[i][0] > wc0[0] and
                    SE_wc[i][1] < wc0[1] and
                    SE_wc[i][1] > wc04[1]):
                    
                    wc04_ind = np.where(wc_x == SE_wc[i][0])[0][0]
                    wc04 = SE_wc[i]
            
            # 3.2.2.1 - found a valid wc04 to swap with
            if (wc04_ind != -1):
                temp_config = copy.deepcopy(wc_pos)
                # replaced with updated 3.2.2.1 logic
                temp_config[wc0_ind, 1] = wc04[1]
                temp_config[wc01_ind, 1] = wc0[1]
                temp_config[wc04_ind, 1] = wc01[1]
                
                if wc01_ind == wc0_ind:
                    raise Exception("Stage 2 Case 3.2.2.1 wc0 and wc01 are the same")
                
                res_boards.append(temp_config)
                res_mults.append(1)
                res_cases.append("Stage 2 Case 3.2.2.1")
            
            #
            wc04_ind = -1
            wc04 = [2*n,-1]
        
            # 3.2.2.2 - seeking a valid wc04 to swap with - highest such wc
            if (row_i < n-1):
                for j in range(len(SW_wc)):
                    if (SW_wc[j][0] > symm_ind(row_i, n)):
                        wc04_ind = np.where(wc_x == SW_wc[j][0])[0][0]
                        wc04 = SW_wc[j]
            
            if (wc04_ind != -1) :
                if row_i < n-1 and wc04_ind != -1:
                    temp_config = copy.deepcopy(wc_pos)
                    
                    # move left 0 to 04 symmetric column
                    temp_config[wc0_ind, 1] = symm_ind(wc04[1], n)
                    # swap 01 and 04
                    temp_config[wc01_ind, 1] = wc0[1]
                    temp_config[wc04_ind, 1] = wc01[1]
                    
                    res_boards.append(temp_config)
                    res_mults.append(1)
                    res_cases.append("Stage 2 Case 3.2.2.2")
                    
    else:
        return Exception("Invalid stage passed to case 3")
    
    return res_boards, res_mults, res_cases

############################################################################################################## END CASE MOVES

# bc_pos and wc_pos are arrays of (r,c) ordered symm. pairs, is either 1 or 2
# desc, asc handle moving rows/columns (along with taking symm pairs) 

def get_wc_move(bc_pos, bc_pos_next, wc_pos, desc, asc, n, stage):
    
    wc_x = wc_pos[:,0]
    wc_y = wc_pos[:,1]
    
    # Determines whether or not this move breaks into multiple cases. Cases returned as strings for verification
    boards = []
    mults = []
    cases = []
    
    desc = np.array(desc, dtype=int)
    asc = np.array(asc, dtype=int)
    
    desc_rows = desc[:,0]
    asc_rows = asc[:,0]
    
    unhappy_ind = np.array([], dtype=int)
    desc_row_ind = np.array([], dtype=int)
    asc_row_ind = np.array([], dtype=int)
    asc_match_ind = np.array([], dtype=int)
    
    # Gets unhappy checkers, finds white checkers on asc, desc rows 
    for i in range(0, len(wc_x)):
        happiness, reasons = check_happy(bc_pos_next, wc_pos, wc_pos[i], n)
        if(not happiness):
            unhappy_ind = np.append(unhappy_ind, i)
            if (wc_x[i] in asc_rows):
                asc_match_ind = np.append(asc_match_ind, i)
        if(wc_x[i] in desc_rows):
            desc_row_ind = np.append(desc_row_ind, i)
        if(wc_x[i] in asc_rows):
            asc_row_ind = np.append(asc_row_ind, i)
            
    # Checkers on consecutive moving rows for case 2
    consec = []
    partner = []
    for s in desc_row_ind:
        for t in asc_row_ind:
            if (wc_x[s] + 1 == wc_x[t]):
                consec.append(s)
                partner.append(t)
                # print("FOUND CONSEC + PARTNER at ", consec, partner)
                
    #Stages don't matter for cases 0,1,2 so those are done first
    if len(unhappy_ind) == 0:
        # CASE 0: where all are happy. Do nothing in this case.
        boards.append(wc_pos)
        mults.append(1)
        cases.append("Case 0: All checkers happy after black checker move")
    elif desc_rows[0] == n-1:
        # CASE 0.5: move on horizontal, done before case 1 since 1 checker may want to ascend
        tbs, tms, tcs = get_horiz_move(bc_pos, bc_pos_next, wc_pos, n)
        num_boards = len(tbs)
        if (num_boards == len(tms) and num_boards == len(tcs)):
            for i in range(0, num_boards):
                boards.append(tbs[i])
                mults.append(tms[i])
                cases.append(tcs[i])
        else:
            return Exception("Move on horizontal, number of boards and number of mults mismatch")
    elif ((len(desc_row_ind) == 0 ) and (len(asc_row_ind) > 0)):
        
        # CASE 1: where moving up as a result of wc on asc row(s)
        tb, tm = get_case1_move(wc_pos, asc_match_ind)
        boards.append(tb)
        mults.append(tm)
        cases.append("Case 1: Moving up as a result of white checkers on ascending row(s)")
        
    elif (len(consec) > 0):
        # CASE 2: where on consecutive moving rows
        # Swap move
        tb, tm = get_case2_move(wc_pos, consec, partner)
        boards.append(tb)
        mults.append(tm)
        cases.append("Case 2: Swap resulting from white checkers on consecutive rows")
        
    else:
        tbs, tms, tcs = get_case3_move(bc_pos, bc_pos_next, wc_pos, desc, asc, n, stage)
        num_boards = len(tbs)
        if (num_boards == len(tms) and num_boards == len(tcs)):
            for i in range(0, num_boards):
                boards.append(tbs[i])
                mults.append(tms[i])
                cases.append(tcs[i])
        else:
            return Exception("Case 3 move, mismatch on boards, mults & cases")
    
    return boards, mults, cases