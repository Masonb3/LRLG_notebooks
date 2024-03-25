import numpy as np
import copy

from LRLG_sym_wc_functions import *

# Takes in a QF table and multiplicity table (both output from rec_white_checker_game)
# and prints out the Q_function expression 

def print_qf(qf_table, mult_table):
    branches = len(qf_table)
    QF_STRING = ""
    for i in range(branches):
        final_degree = mult_table[i][-1]
        
        nontrivial_term = False
        qf_term = str(final_degree) + "Q_{"
        
        for j in range(len(qf_table[i])):
            ind = qf_table[i][j]
            if (ind) > 0:
                qf_term += str(ind)
                # as long as the Q function has at least one index, we want it to show up in the expression
                nontrivial_term = True
        if (i != branches - 1):
            qf_term += "}   +   "
        else:
            qf_term += "}"
        
        if nontrivial_term: 
            QF_STRING += qf_term
    
    print(QF_STRING)
    
    
def degen_print_qf(qf_set, mult_set):
    branches = len(qf_set)
    QF_STRING = ""
    for i in range(branches):
        final_degree = mult_set[i]
        
        nontrivial_term = False
        qf_term = str(final_degree) + "Q_{"
        
        for j in range(len(qf_set[i])):
            ind = qf_set[i][j]
            if (ind) > 0:
                qf_term += str(ind)
                # as long as the Q function has at least one index, we want it to show up in the expression
                nontrivial_term = True
        if (i != branches - 1):
            qf_term += "}   +   "
        else:
            qf_term += "}"
        
        if nontrivial_term: 
            QF_STRING += qf_term
    
    print(QF_STRING)
    
# Recursively degenerates based on white checker position from particular step,
# Note: 1 <= step <= n^2
def degenerate(wc_pos, step, n, mult_set, partition_set, branch_label=[1], mult=1, partition=[], togglePrints=True):
    
    Npos, De, As = get_ngame_bc_pos(n)
    
    print("START STEP ", step-1, "to", step, " IN ", n)
    bc_p = Npos[step-1]
    bc_pn = Npos[step]
    bc_de = De[step-1]
    bc_as = As[step-1]
    
    # Hardcoded, stage 2 begins after one checker has crossed the horizontal midline,
    # Which takes (n-1) + ... + 1 moves, and we index at 1, so we add 1 here
    stage = 1
    if (n != 1) and (step > int(((n)*(n-1))/2) + 1):
        stage = 2

    res_boards, res_mults, res_cases = get_wc_move(bc_p, bc_pn, wc_pos, bc_de, bc_as, n, stage)
    
    num_boards = len(res_boards)
    
    # Error checking
    if (num_boards != len(res_mults)) or (num_boards != len(res_cases)):
        raise Exception("DEGEN: Mismatch in length of boards/mults/cases : ", num_boards, "/", len(res_mults), "/", len(res_cases) )

    # Printing current results and updating various variable
    for i in range(0, num_boards):
        print("PRINTING BOARD ", i, " FOR STEP ", step, " WITH N=", n)
        print("BOARD IS", res_boards[i])
        print("OLD BB: ", bc_p, "\nNEW BB: ", bc_pn)
        cur_board = res_boards[i]
        cur_mult = mult * res_mults[i]
        c_branch_label = copy.deepcopy(branch_label)    
        c_partition = copy.deepcopy(partition)
        
        # Updating branch if necessary
        if(num_boards > 1):
            c_branch_label.append(i+1)
           
        if (togglePrints):
            print("STEP ", step-1, " -> ", step, " IN N=", n)
            print_board(add_boards(pos_to_board(bc_pn, n, "X"), pos_to_board(cur_board, n, "O")))
            print("Branch: ", c_branch_label)
            print("Multiplicity: ", cur_mult)
            print("Partition:", c_partition)
            print("Case: ", res_cases[i])
            
        # Decrease for arrival of black checkers at NW, SE corners
        bc_first_col_ind = np.where(bc_pn[:,1] == 0)[0][0]
        bc_fc = bc_pn[bc_first_col_ind]
        if (bc_fc[0] == 0):
            print("POPPED!")
            # Send smaller board and update partition value accordingly
            popped, partition_val = pop_board(cur_board, n) 
            cur_board = popped
            # Update step, n for next iteration
            n -= 1
            if n == 1:
                step = 1
            elif n > 1:
                step = int(((n)*(n-1))/2) + 1
                
            c_partition.append(partition_val)
        else:
            step += 1
  
        if (n != 0):
            print("DEGENERATE S/N", step, n)
            degenerate(cur_board, step, n, mult_set, partition_set, c_branch_label, cur_mult, c_partition, togglePrints=True)
        else: 
            # Record the partition at the very end of the degeneration
            print("RECORDING PARTITION SET/MULT SET")
            print("PRE PS", partition_set)
            partition_set.append(c_partition)
            mult_set.append(cur_mult)
            print("PARTITION SET", partition_set)

# Does a full white checker game based on the recursive degeneration given above
def degen_game(lam, mu, n, ms=[], ps=[], togglePrints=True):
    wc_pos_init = init_white_config(lam,mu,n)
    
    BC, tempD, tempA = get_ngame_bc_pos(n) 
    if(togglePrints):
        print("STEP 0 IN N=", n)
        print_board(add_boards(pos_to_board(BC[0], n, "X"), pos_to_board(wc_pos_init, n, "O")))
        print("Branch: [1]")
        print("Degree: [1]")
        print("Partition: []")
    
    degenerate(wc_pos_init, 1, n, ms, ps, togglePrints=togglePrints)
    if(togglePrints):
        print("QF RESULT:")
        print(ps, ms)
        degen_print_qf(ps, ms)
    
# Takes in partitions and n for multiplication, performs 
# white checker moves until a final board is reached for each branch.
def rec_white_checker_game(lam, mu, n, branch=[], togglePrints=False):
    steps = n**2 + 1
    
    Npos, De, As = get_ngame_bc_pos(n)
    wc_pos_init = init_white_config(lam,mu,n)
    
    degs_init = []
    for w_ind in range(len(wc_pos_init)):
        deg = get_wc_degree(w_ind, Npos[0], wc_pos_init, n)
        degs_init.append(deg)
    
    # HOLDS ALL WC_BOARDS ACROSS GAMES, 
    #     FIRST INDEX IS BRANCH, SECOND IS STEP
    # NOTE THAT FOR EACH BOARD, THERE WILL BE A SUBSET
    # OF WHITE CHECKERS CORRESPONDING TO SUBBOARD THAT
    # IS CONSIDERED
    wc_board_sets = [[np.array(wc_pos_init,dtype=int)]]
    mult = [[1]]
    qf_ind_sets = [[-1]]
    deg_sets = [[degs_init]]
    
    # indexes over steps of boards - first board not included because it does not
    # come from a previous board
    for i in range(0,steps-1):
        
        stage = 1
        
        zero_index = np.where(Npos[i][:,1] == 0)[0][0]
        
        sub_count = n
        if Npos[i][zero_index][0] < n:
            stage = 2
            
            removing = True
            diag_col = 0
            while removing:
                bc_dc_ind = np.where(Npos[i][:,1] == diag_col)[0][0]
                bc_dc = Npos[i][bc_dc_ind]
                # print("BC_DC: ", bc_dc)
                if (bc_dc[0] == diag_col):
                    sub_count -= 1
                    diag_col += 1
                else:
                    removing = False
        
        if (i == 0 and togglePrints):
            
            bcboard = pos_to_board(Npos[0], n, "X")
            wcboard = pos_to_board(wc_pos_init, n, "O")
    
            print("Board at step 1:")
            print_board(add_boards(bcboard, wcboard))
            first_degs = []
            for w_ind in range(len(wc_pos_init)):
                one_first_deg = get_wc_degree(w_ind, Npos[0], wc_pos_init, n)
                first_degs.append(one_first_deg)
                #print("WC ordering check: ", wc_pos)
            print("Degrees for white checkers ", first_degs, "\n \n")
        
        # indexes over branches
        for j in range(0, len(wc_board_sets)):
            
            print("IN ATTEMPT TO PRINT BRANCH " + str(j) + " AT STEP " + str(i))
            wc_old_sub = wc_board_sets[j][i]
            mult_old = mult[j][i]
            
            sub_count_old = len(wc_board_sets[j][i])
            
            # if we have decreased, we need to use the old board and change just a bit
            if (sub_count < sub_count_old):
                # First determine whether a white checker ended in the top spot or the bottom
                if 0 in wc_old_sub[:,1]:
                    qf_ind_sets[j].append(sub_count+1)
                elif 2*sub_count_old - 1 in wc_old_sub[:,1]:
                    # NOTE: 2n+1 here instead of 2n-1 since sub_count is updated before this check
                    qf_ind_sets[j].append(-1)
                else: 
                    raise Exception("Reducing to subboard, but no white checker in either of the two corner spots.") 
                
                # Then update so that we are only considering the sub-board in the future
                wc_old_sub = np.array(get_sub_board(wc_old_sub, sub_count, sub_count_old))
                    
            bc_old_sub = np.array(get_sub_board(Npos[i], sub_count, n))
            bc_new_sub = np.array(get_sub_board(Npos[i+1], sub_count, n))
            
            desc_sub = np.array(get_sub_board(De[i], sub_count, n))
            asc_sub = np.array(get_sub_board(As[i], sub_count, n))
    
            # weird fix for final qf index 
            if i == steps - 2:
                if (wc_old_sub[0][0] == 1 and wc_old_sub[0][1] == 1):
                    qf_ind_sets[j].append(-1)
                else:
                    qf_ind_sets[j].append(1)
            wcon, inc_mults, cases = get_wc_move(bc_old_sub, bc_new_sub, wc_old_sub, desc_sub, asc_sub, sub_count, stage)
            
            # Do this before appending the new board to copy the old branch for splitting
            if (len(wcon) > 1):
                old_branch = copy.deepcopy(wc_board_sets[j])
                wc_board_sets.append(old_branch)
                
                old_mult = copy.deepcopy(mult[j])
                mult.append(old_mult)
                
                old_degs = copy.deepcopy(deg_sets[j])
                deg_sets.append(old_degs)
                
                old_qf_ind = copy.deepcopy(qf_ind_sets[j])
                qf_ind_sets.append(old_qf_ind)

                # Now we can add the new branch at the (j+1)th spot
                wc_board_sets[j+1].append(wcon[1])

                mult[j+1].append(inc_mults[1]*mult_old)
                
                degs_new = []
                for w_ind in range(len(wcon[1])):
                    deg = get_wc_degree(w_ind, bc_new_sub, wcon[1], n)
                    degs_new.append(deg)
                deg_sets[j+1].append(degs_new)
                
                # I don't think any updates need to happen for qf indices, since splits
                # can't happen when going to corners? could be wrong

            # And then we can add the move from the original branch at the old spot,
            #     which happens regardless of the split.

            wc_board_sets[j].append(wcon[0])
            mult[j].append(inc_mults[0]*mult_old)
            
            degs_new = []
            for w_ind in range(len(wcon[0])):
                deg = get_wc_degree(w_ind, bc_new_sub, wcon[0], n)
                degs_new.append(deg)
            deg_sets[j].append(degs_new)
            
            # NEED TO FIX SO THIS SHOWS BRANCHES BETTER
            if togglePrints:
                for k in range(len(cases)):
                    bcboard = pos_to_board(bc_new_sub, sub_count, "X")
                    wcboard = pos_to_board(wcon[k], sub_count, "O")

                    degprint = []
                    for w_ind in range(len(wcon[k])):
                        deg = get_wc_degree(w_ind, bc_new_sub, wcon[k], n)
                        degprint.append(deg)
                    
                    print("Board at step " + str(i+2) + ":")
                    print("   Branch " + str(j + k) + " case " + cases[k])
                    print_board(add_boards(bcboard, wcboard)) 
                    print("Current multiplicity is ", mult[j+k][i+1])
                    print("Degrees for white checkers ", degprint, "\n ")
                    
            
    return wc_board_sets, mult, deg_sets, qf_ind_sets
    # Fix branching to use less memory - make into a tree rather than so many copies of the boards

# Combines QF functionality and recursive game functionality to produce a game
# With corresponding printouts of expected QF terms. 
def full_game(lam, mu, n, tPgame=False,tPqf=False):
    full_boards, full_mult, full_degs, full_qf = rec_white_checker_game(lam, mu, n, togglePrints=tPgame)
    if (tPqf):
        print_qf(full_qf, full_mult)
    return full_boards, full_mult, full_degs, full_qf