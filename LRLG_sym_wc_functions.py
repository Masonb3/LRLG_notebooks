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
#    wc_curr should be [x,y] of the wc in question
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
            #print("BC ", bc_x[i], bc_y[i], " vs. ", x,y)
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
        total_deg += get_wc_degree(i, bc_pos, wc_pos, n)
        
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
    
    bc_x = bc_pos[:,0]
    bc_y = bc_pos[:,1]
    
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
        
        a = np.where(bc_x == wc0[0])[0][0]
        b = np.where(bc_x == wc1[0])[0][0]
        
        bc0 = bc_pos[a] 
        bc1 = bc_pos[b]
        
        #bc0 = bc_pos[bc_x.index(wc0[0])]
        #bc1 = bc_pos[bc_x.index(wc1[0])]
        
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

# bc_pos and wc_pos are arrays of (r,c) ordered symm. pairs, stage is either 1 or 2
# desc, asc handle moving rows/columns (along with taking symm pairs) 

def get_wc_move(bc_pos, bc_pos_next, wc_pos, desc, asc, n, stage):
    
    bc_x = bc_pos[:,0]
    bc_y = bc_pos[:,1]
    
    bc_x_next = bc_pos_next[:,0]
    bc_y_next = bc_pos_next[:,1]
    
    wc_x = wc_pos[:,0]
    wc_y = wc_pos[:,1]
    
    # Determines whether or not this move breaks into multiple cases. Cases returned as strings for verification
    split = False
    cases = []
    mults = np.array([], dtype=int)
    
    desc = np.array(desc, dtype=int)
    asc = np.array(asc, dtype=int)
    
    desc_rows = desc[:,0]
    asc_rows = asc[:,0]
    
    wc_new_config_x = copy.deepcopy(wc_x)
    wc_new_config_y = copy.deepcopy(wc_y)
    
    wc_split_config_x = copy.deepcopy(wc_x)
    wc_split_config_y = copy.deepcopy(wc_y)
    
    unhappy_ind = np.array([], dtype=int)
    desc_row_ind = np.array([], dtype=int)
    asc_row_ind = np.array([], dtype=int)
    desc_row_match = np.array([], dtype=int)
    asc_row_match = np.array([], dtype=int)
    
    OLD_TOTAL_DEG = get_total_wc_degree(bc_pos, wc_pos, n)
    
    for i in range(0, len(wc_x)):
        happiness, reasons = check_happy(bc_pos_next, wc_pos, wc_pos[i], n)
        if(not happiness):
            unhappy_ind = np.append(unhappy_ind, i)
        if(wc_x[i] in desc_rows):
            desc_row_ind = np.append(desc_row_ind, i)
            desc_row_match = np.append(desc_row_match, wc_x[i])
        if(wc_x[i] in asc_rows):
            asc_row_ind = np.append(asc_row_ind, i)
            asc_row_match = np.append(asc_row_match, wc_x[i])
            
    consec = -1
    partner = -1
    for s in desc_row_ind:
        for t in asc_row_ind:
            if (wc_x[s] + 1 == wc_x[t]):
                consec = s
                partner = t
                # print("FOUND CONSEC + PARTNER at ", consec, partner)
                
    #Stages don't matter for cases 0,1,2 so those are done first
    if len(unhappy_ind) == 0:
        # CASE 0: where all are happy. Do nothing in this case.
        cases.append("Stage " + str(stage) + " Case 0")
        mults = np.append(mults, 1)
    
    elif ((len(desc_row_ind) == 0 ) and (len(asc_row_ind) > 0)):
        
        # CASE 1: where moving up as a result of wc on asc row(s)
        for k in asc_row_ind:
            wc_new_config_x[k] -= 1
        cases.append("Stage " + str(stage) + " Case 1")
        mults = np.append(mults, 1)
        
    elif (consec != -1):
        # CASE 2: where on consecutive moving rows
        # Swap move
        wc_new_config_y[s] = wc_y[t]
        wc_new_config_y[t] = wc_y[s]
        cases.append("Stage " + str(stage) + " Case 2")
        mults = np.append(mults, 1)
    elif desc_rows[0] == n-1 and n == 1:
        # only one white checker, and if it is in bottom right, it stays, otherwise it moves to top left
        if (wc_pos[0][0] == n and wc_pos[0][1] == n):
            wc_new_config_y[0] = n
            cases.append("n=1, ended in bottom right")
            mults = np.append(mults, 1)
        else: 
            wc_new_config_y[0] = 0
            cases.append("n=1, ended in top left")
            mults = np.append(mults, 1)
            
    elif desc_rows[0] == n-1 and len(desc_row_ind) > 0:
        # What happens when moving across horizontal?
        # I think split one for going down and one for going to left
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
        horiz_move_config[wc_onr_ind][1] = 0
        test_move_deg = get_total_wc_degree(bc_pos_next, horiz_move_config, n)
        if test_move_deg == OLD_TOTAL_DEG:
            move_search = False
            moved_far = True
        
        while move_search and search_count < len(sw_to_check):
            sw_wc_curr = sw_to_check[search_count]
            print("Trying to put ", sw_wc_curr, " to right during horizontal move.")
            sw_wc_curr_ind = np.where(wc_y == sw_wc_curr[1])[0][0]
            horiz_move_config[sw_wc_curr_ind][1] = symm_ind(sw_wc_curr[1], n)
            horiz_move_config[wc_onr_ind][1] = 0
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
        print("WC_POS, n", wc_pos, n, "\n")
        print("GOT RIGHT SE CHECKER: ", se_to_swap)
        
        horiz_stay_config[se_to_swap_ind][1] = 2*n-1
        horiz_stay_config[wc_onr_ind][1] = se_to_swap[1]
        
        test_stay_deg = get_total_wc_degree(bc_pos_next, horiz_stay_config, n)
        if test_stay_deg == OLD_TOTAL_DEG: 
            stayed = True
        else: 
            print("HERE'S THE FAILING STAY CONFIG")
            horiz_stay_board = pos_to_board(horiz_stay_config, n, "O")
            bc_stay_board = pos_to_board(bc_pos_next, n, "X")
            print_board(add_boards(horiz_stay_board, bc_stay_board))
            print("OLD_TOTAL_DEG", OLD_TOTAL_DEG, "\nTEST_STAY_DEG", test_stay_deg)
        
        if moved_far:
            cases.append("Far move on horizontal cross")
            mults = np.append(mults, 2)
            wc_new_config_y = horiz_move_config[:,1]
            
            if stayed:
                cases.append("Stay move on horizontal cross")
                mults = np.append(mults, 1)
                wc_split_config_y = horiz_stay_config[:,1]
                split = True
        else: 
            if stayed:
                cases.append("Stay move on horizontal cross")
                mults = np.append(mults, 1)
                wc_new_config_y = horiz_stay_config[:,1]
            else:
                raise Exception("W/C did not move properly at horizontal cross")            
# STAGE 1 CASE 3 
    elif stage == 1:
        # CASE 3: Awc = Attached Checker
        row_i = min(desc[0][0], desc[1][0])
        col_j = min(desc[0][1], desc[1][1])
        # MIGHT NEED TO FIX COL_J
        
        l = col_j - min(asc[0][1], asc[1][1]) 
        
        if (row_i not in wc_x) or wc_pos[np.where(wc_x == row_i)[0], 1] != bc_pos[np.where(bc_x == row_i)[0], 1]:
            # Shouldn't happen as given in cases
            raise Exception("In Stage 1 Case 3, but no attached white checker was found at i=" + str(row_i))
        else:
            # GET POSITIVE I from desc black checkers
            if row_i < n-1:
                opp_im = symm_ind(board_ind(inv_board_ind(row_i,n) - 1, n), n) 
                
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
                        wc_new_config_y[sad_NE_ind] -= 1
                        wc_new_config_y[swap_3_1_ind] = on_col_j_wc[1]
                        wc_new_config_y[on_col_j_ind] = col_j - l
                        
                        cases.append("Stage 1 Case 3.1 - Small move variant")
                        
                    else:
                        moved_left = get_move_left(bc_pos_next, wc_pos, [np.where(wc_x == row_i)[0][0], np.where(wc_y == col_j)[0][0]], n)
                        wc_new_config_x = moved_left[:,0]
                        wc_new_config_y = moved_left[:,1]
                        
                        cases.append("Stage 1 Case 3.1 - Large move variant")
                    
                    
                    mults = np.append(mults, 1)
                    
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
                    wc_new_config_y[sw_ind] = wc0[1]
                    wc_new_config_y[wc0_ind] = sw_j
                
                    cases.append("Stage 1 Case 3.2.1")
                    mults = np.append(mults, 1)
                        
                    # CASE 3.2.2.
                        # Shift columns, checking to see if there is a checker that can be shifted
                        # after moving O to the right by 1
                    s0 = wc0[1] - 1
                    
                    if(symm_ind(s0, n) in wc_y):
                        split = True
                        
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
                        
                        wc_split_config_y[wc02_ind] = s1 - 1 
                        wc_split_config_y[wc01_ind] = s2
                        wc_split_config_y[wc0_ind] = s0
                        
                        cases.append("Stage 1 Case 3.2.2")
                        if (col_j + l) == n and (wc01[0] == symm_ind(row_i + 1, n)):
                            mults = np.append(mults, 2)
                        else:
                            mults = np.append(mults, 1)
                        
    elif stage == 2:
        row_i = min(desc[0][0], desc[1][0])
        
        moving_col = np.array([desc[0][1], desc[1][1], asc[0][1], asc[1][1]])
        moving_col = np.sort(moving_col)
        
        # idea here is that moving_col sorted will be of form {n, j, -j, -n}
        col_j = moving_col[1]
        
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
        
        if blocked:
            # Case 3.1
            wc03_ind = np.where(wc_x == wc03[0])[0][0]
                
            print("wc03 is at ", wc03)
            print("wc0 is at ", wc0)
            print("wc01 is at ", wc01)
            wc_new_config_y[wc0_ind] = wc03[1]
            # NEW 3.1 ADDED AFTER FAILURE AT [2,1]*[2]
            wc_new_config_y[wc03_ind] = wc01[1]
            wc_new_config_y[wc01_ind] = wc0[1]
            cases.append("Stage 2 Case 3.1")
            mults = np.append(mults, 1)
            
#             wc_new_config_y[wc03_ind] = wc0[1]

                
#             # In new configuration! Make sure this is happening with wc_new_config
#             wc_new_config_temp = np.zeros((len(wc_x), 2), dtype = int)
#             wc_new_config_temp[:,0] = wc_new_config_x
#             wc_new_config_temp[:,1] = wc_new_config_y
                
#             SE_wc = get_SE(wc_new_config_temp, n)
#             # must be a SE checker SW of 01
#             SE_to_swap_ind = -1
#             SE_to_swap = [-1,2*n]
#             for j in range(0, len(SE_wc)):
#                 if(SE_wc[j][0] > wc01[0] and 
#                    SE_wc[j][1] < wc01[1] and
#                    SE_wc[j][1] < SE_to_swap[1]):
#                     SE_to_swap_ind = j
#                     SE_to_swap = [SE_wc[j][0], SE_wc[j][1]]
                
#             if SE_to_swap_ind != -1:
#                 wc_new_config_y[wc01_ind] = SE_to_swap[1]
#                 wc_new_config_y[SE_to_swap_ind] = wc01[1]
#                 cases.append("Stage 2 Case 3.1")
#                 mults = np.append(mults, 1)
                
#             else:
#                 raise Exception("In Case 3.1 failed to swap 01")
                    
        elif not blocked:
            # Split cases
            
            # Case 3.2.1
            wc_new_config_y[wc0_ind] = 0
                
            wc_new_config_temp = np.zeros((len(wc_x), 2), dtype = int)
            wc_new_config_temp[:,0] = wc_new_config_x
            wc_new_config_temp[:,1] = wc_new_config_y
                
            made_02_happy = False
            made_01_happy = False
            
            new_02_col = wc02[1] + 1
            new_01_col = wc01[1] - 1
            
            if (wc02_ind != -1):
                while not made_02_happy and new_02_col < 2*n:
                    wc02_test = [wc02[0], new_02_col]
                    # changing wc_new_config_temp to make check_happy not test against wc02_test
                    wc_new_config_temp[wc02_ind] = wc02_test
                    is_hap, r = check_happy(bc_pos_next, wc_new_config_temp, wc02_test, n)
                    if is_hap:
                        made_02_happy = True
                        wc_new_config_y[wc02_ind] = new_02_col
                    else:
                        new_02_col += 1
            else:
                made_02_happy = True
            
            while not made_01_happy and new_01_col >= 0:
                wc01_test = [wc01[0], new_01_col]
                # changing wc_new_config_temp to make check happy not test against wc01_test
                wc_new_config_temp[wc01_ind] = wc01_test
                #print ("NEW CONFIG TEMP", wc_new_config_temp)
                is_hap, r = check_happy(bc_pos_next, wc_new_config_temp, wc01_test, n)
                if is_hap:
                    made_01_happy = True
                    wc_new_config_y[wc01_ind] = new_01_col
                    #print(" MADE 01 HAPPY from ", wc01[1], " to ", new_01_col)
                else:
                    new_01_col -= 1
                        
            if not made_02_happy:
                raise Exception("Case 3.2.1 could not move 02 right to be happy. wc0 at ", wc0, " wc01 at ", wc01, " wc02 at ", wc02)
            elif not made_01_happy:
                raise Exception("Case 3.2.1 could not move 01 left to be happy. 01 at ", wc01)
            else:
                cases.append("Stage 2 Case 3.2.1")
                mults = np.append(mults, 1)
# Previously used this, but I think 3.2.1 does not accrue multiplicity, maybe change if it isn't correct
#                 if wc02_ind != -1:
#                     mults = np.append(mults, 2)
#                 else: 
#                     mults = np.append(mults, 1)
                
                    
            # Case 3.2.2
                # first finding if any checkers in SE are SW of wc0
            wc04_ind = -1
            wc04 = [-1,-1]
        
            for i in range(0, len(SE_wc)):
                if (SE_wc[i][0] > wc0[0] and
                    SE_wc[i][1] < wc0[1] and
                    SE_wc[i][1] > wc04[1]):
                    
                    wc04_ind = np.where(wc_x == SE_wc[i][0])[0][0]
                    wc04 = SE_wc[i]
            
            # found a valid wc04 to swap with
            if (wc04_ind != -1):
                # replaced with updated 3.2.2.1 logic
                wc_split_config_y[wc0_ind] = wc04[1]
                wc_split_config_y[wc01_ind] = wc0[1]
                wc_split_config_y[wc04_ind] = wc01[1]
                
                if wc01_ind == wc0_ind:
                    raise Exception("Stage 2 Case 3.2.2.1 wc0 and wc01 are the same")
                
#                 wc_split_config_temp = np.zeros((len(wc_x), 2), dtype = int)
#                 wc_split_config_temp[:,0] = wc_split_config_x
#                 wc_split_config_temp[:,1] = wc_split_config_y
                
#                 # looking for checker SW of wc01 to swap with:
#                 wc_sw01_ind = -1
#                 wc_sw01 = [-1,-1]
#                 for i in range(0, len(wc_split_config_x)):
#                     if (wc_split_config_temp[i][0] > wc01[0] and
#                         wc_split_config_temp[i][1] < wc01[1] and
#                         wc_split_config_temp[i][1] > wc_sw01[1]):
                    
#                         wc_sw01_ind = i
#                         wc_sw01 = wc_split_config_temp[i]
                    
#                 # Not gauranteed to occur, but will cause a split
#                 if wc_sw01_ind != -1:
#
#                     wc_split_config_y[wc01_ind] = wc_sw01[1]
#                     wc_split_config_y[wc_sw01_ind] = wc_sw01[1]
                # Removed above because I think it's not necessary as long as wc01 always exists properly
                cases.append("Stage 2 Case 3.2.2.1")
                mults = np.append(mults, 1)
                    
                split = True
            else:
                wc04_ind = -1
                wc04 = [-1,-1]
                
                for j in range(len(SW_wc)):
                    if (SW_wc[j][0] > symm_ind(row_i, n)):
                        wc04_ind = np.where(wc_x == SW_wc[j][0])[0][0]
                        wc04 = SW_wc[j]
                
                if row_i < n-1 and wc04_ind != -1:
                    # move left 0 to 04 symmetric column
                    wc_split_config_y[wc0_ind] = symm_ind(wc04[1], n)
                    # swap 01 and 04
                    wc_split_config_y[wc01_ind] = wc04[1] - 1
                    wc_split_config_y[wc04_ind] = wc01[1]
                    
                    split = True
                    
                    cases.append("Stage 2 Case 3.2.2.2")
                    mults = np.append(mults, 1)
                    
    else:
        raise Exception("Failed to identify Stage/Case.")
    
    wc_new_config = np.zeros((len(wc_x), 2), dtype = int)
    wc_new_config[:,0] = wc_new_config_x
    wc_new_config[:,1] = wc_new_config_y
    
    wc_new_configs = [wc_new_config]
    
    if split:
        wc_split_config = np.zeros((len(wc_x), 2), dtype = int)
        wc_split_config[:,0] = wc_split_config_x
        wc_split_config[:,1] = wc_split_config_y
        
        wc_new_configs.append(wc_split_config)
        
    return split, wc_new_configs, cases, mults