import os
import numpy as np
import copy

def load_sudoku(puzzle_path):
    ''' Load the sudoku from the given path; it returns the sudoku as a list of lists
        input: puzzle_path: path to the puzzle
        output: ret: the sudoku as a list of lists where -1 represents an empty cell
        and 0-8 represents the value in the cell corresponding to the numbers 1-9'''
    ret = []
    with open(puzzle_path, 'r') as sudoku_f:
        for line in sudoku_f:
            line = line.rstrip()
            cur = line.split(" ")
            cur = [ord(x) - ord('1') + 1 if x.isdigit() else -1 for x in cur]
            ret.append(cur)
    print(ret)
    return ret

def get_neighbors(xi):
    lst = []
    for i in range(9):
        for j in range(9):
            lst.append((i, j))

    f_lst = {}
    for l in lst:
        lst_in = []
        for i in range(9):
            if i != l[1]:
                lst_in.append((l[0], i))
        for i in range(9):
            if i != l[0]:
                lst_in.append((i, l[1]))
        for i in range((xi[0] // 3) * 3, ((xi[0] // 3) * 3)+3):
            for j in range((xi[1] // 3) * 3, ((xi[1] // 3) * 3)+3):
                if (i, j) not in lst_in:
                    lst_in.append((i, j))
        f_lst[l] = (lst_in)

    return f_lst[xi]

def isSolved(sudoku):
    '''' Check if the sudoku is solved
        input: sudoku: the sudoku to be solved
               kwargs: other keyword arguments
        output: True if solved, False otherwise
    '''
    for i in range(len(sudoku)):
        for j in range(len(sudoku)):
            if sudoku[i][j] == -1:
                return False
    return True

def undo_changes_for_position(sudoku, x, y, val):
    ''' Undo the changes made for the given position
        input: sudoku: the sudoku to be solved
               x: row number
               y: column number
               val: value to be checked
               kwargs: other keyword arguments
        output: None
    '''
    sudoku[x][y] = -1


def update_changes_for_position(sudoku, x, y, val):
    ''' Update the changes for the given position
        input: sudoku: the sudoku to be solved
               x: row number
               y: column number
               val: value to be checked
               kwargs: other keyword arguments
        output: None
    '''
    sudoku[x][y] = val


def isPossible(sudoku, x, y, val):
    ''' Check if the value(val) is possible at the given position (x, y)
        input: sudoku: the sudoku to be solved
               x: row number
               y: column number
               val: value to be checked
               kwargs: other keyword arguments
        output: True if possible, False otherwise
    '''
    for i in range(len(sudoku)):
        if sudoku[x][i] == val:
            return False
        if sudoku[i][y] == val:
            return False

    for i in range((x // 3) * 3, ((x // 3) * 3)+3):
        for j in range((y // 3) * 3, ((y // 3) * 3)+3):
            if sudoku[i][j] == val:
                return False

    return True

def get_mrv_position(sudoku, dms):
    ''' Get the position with minimum remaining values
        input: sudoku: the sudoku to be solved
               kwargs: other keyword arguments'
        output: x: row number
                y: column number'''
    dms = get_domain(sudoku)
    x, y, min_dms = -1, -1, float('inf')

    for i in range(len(sudoku)):
        for j in range(len(sudoku)):
            if sudoku[i][j] == -1:
                domain_size = len(dms[i][j])
                if domain_size < min_dms:
                    min_dms = domain_size
                    x = i
                    y = j
    return x, y


def get_domain(sudoku):
    domains = []
    for i in range(9):
        domain_row = []
        for j in range(9):
            if sudoku[i][j] == -1:
                domain = list(range(1, 10))
                for k in range(len(sudoku)):
                    if sudoku[i][k] in domain:
                        domain.remove(sudoku[i][k])
                for k in range(len(sudoku)):
                    if sudoku[k][j] in domain:
                        domain.remove(sudoku[k][j])
                box = []
                for k in range((i // 3) * 3, ((i // 3) * 3)+3):
                    for l in range((j // 3) * 3, ((j // 3) * 3)+3):
                        box.append(sudoku[k][l])
                for k in range(len(box)):
                    if box[k] in domain:
                        domain.remove(box[k])
                domain_row.append(domain)
            else:
                domain_row.append([sudoku[i][j]])
        domains.append(domain_row)
    return domains

def undo_waterfall_changes(sudoku, changes, dms):
    ''' Undo the changes made by the waterfalls
        input: sudoku: the sudoku to be solved
               changes: list of changes made by the waterfalls previously
               kwargs: other keyword arguments
        output: None

    '''
    for c in changes:
        dms[c[0][0]][c[0][1]].append(c[1])

def apply_waterfall_methods(sudoku, list_of_waterfalls, dms, arcs):
    ''' Apply the waterfall methods to the sudoku
        input: sudoku: the sudoku to be solved
               list_of_waterfalls: list of waterfall methods
               kwargs: other keyword arguments
        output: isPoss: True if the sudoku is solved, False otherwise
                all_changes: list of changes made by the waterfalls'''
    all_changes = []

    #Keep applying the waterfalls until no change is made
    while True:
        #Flag to check if any change is made by the waterfalls
        any_chage = False
        for waterfall in list_of_waterfalls:
            isPoss, changes = waterfall(sudoku, dms, arcs)
            all_changes += changes
            # If any change is made, then set the flag to True
            if len(changes) > 0:
                any_chage = True
            # If the sudoku is not possible fill up, then return False and the changes made to be undone
            if not isPoss:
                return False, all_changes
        # If no change is made by the waterfalls at current iteration, then break
        if not any_chage:
            break
    return True, all_changes


def get_next_position_to_fill(sudoku, x, y, mrv_on, dms):
    ''' Get the next position to fill durin the backtracking
        input: sudoku: the sudoku to be solved
               x: row number
               y: column number
               mrv_on: True if mrv is on, False otherwise
               kwargs: other keyword arguments
        output: nx: next row number
                ny: next column number'''
    if mrv_on == False:
        for i in range(len(sudoku)):
            for j in range(len(sudoku)):
                if sudoku[i][j] == -1:
                    return i, j
        return -1, -1

    if mrv_on == True:
        return get_mrv_position(sudoku, dms)


def solve_sudoku(sudoku, x, y, mrv_on, list_of_waterfalls, dms, arcs):
    '''' Solve the sudoku using the given waterfall methods with/without mrv
        input: sudoku: the sudoku to be solved
               x: row number of the current position
               y: column number the current position
               mrv_on: True if mrv is on, False otherwise
               list_of_waterfalls: list of waterfalls to be applied
               kwargs: other keyword arguments
               output:  True if solved, False otherwise
                        sudoku: the solved sudoku
                        guess: number of guesses made'''
    #Feel free to change the function as you need, for example, you can change the keyword arguments in the function calls below
    #First you need to check whether the sudoku is solved or not
    if isSolved(sudoku):
        return True, sudoku, 0
    #Apply the waterfalls; change the kwargs with your own
    isPoss, changes = apply_waterfall_methods(sudoku, list_of_waterfalls, dms, arcs)

    # If the sudoku is not possible, undo the changes and return False
    if not isPoss:
        print("Not Is Poss")
        undo_waterfall_changes(sudoku, changes, dms)
        return False, sudoku, 0
    #After waterfalls are applied, now you need to check if the current position is already filled or not; if it is filled,
    #then you need to get the next position to fill

    if sudoku[x][y] != -1:
        nx, ny = get_next_position_to_fill(sudoku, x, y, mrv_on, dms)
        solved, sudoku, guess = solve_sudoku(sudoku, nx, ny, mrv_on, list_of_waterfalls, dms, arcs)

        if solved:
            return True, sudoku, guess
        else:
            #Undo the changes made by the already applied waterfalls
            undo_waterfall_changes(sudoku, changes, dms)
            return False, sudoku, guess



    no_cur_guess = 0
    #Check how many guesses are possible for the current position
    for i in range(1, 10):
        if isPossible(sudoku, x, y, i):
            no_cur_guess += 1

    if no_cur_guess == 0:
        return False, sudoku, 0

    for i in range(1, 10):
        #Check if the value is possible at the current position
        if isPossible(sudoku, x, y, i):
            #If the value is possible, then update the changes for the current position
            update_changes_for_position(sudoku, x, y, i)
            #Get the next position to fill
            nx, ny = get_next_position_to_fill(sudoku, x, y, mrv_on, dms)
            #Solve the sudoku for the next position
            solved, sudoku, guesses = solve_sudoku(sudoku, nx, ny, mrv_on, list_of_waterfalls, dms, arcs)
            no_cur_guess += guesses

            #If the sudoku is solved, then return True, else undo the changes for the current position
            if solved:
                return True, sudoku, no_cur_guess - 1
            else:
                undo_changes_for_position(sudoku, x, y, i)

    #If the sudoku cannot solved at current partially filled state, then undo the changes made by the waterfalls and return False
    undo_waterfall_changes(sudoku, changes, dms)
    return False, sudoku, no_cur_guess - 1

def revise(sudoku, dms, xi, xj):
    revised = False
    to_remove = []
    changes = []
    if len(dms[xj[0]][xj[1]]) == 1:
        for value in dms[xi[0]][xi[1]]:
            if value in dms[xj[0]][xj[1]]:
                revised = True
                to_remove.append(value)
                changes.append((xi, value))
    for i in dms[xi[0]][xi[1]]:
        if i in to_remove:
            dms[xi[0]][xi[1]].remove(i)
    return revised, changes

def ac3_waterfall(sudoku, dms, arcs):
    '''The ac3 waterfall method to apply
    input:  sudoku: the sudoku to apply AC-3 method on'
            kwargs: the kwargs to be passed to the isPossible function
    output: isPoss: whether the sudoku is still possible to solve (i.e. not inconsistent))
            changes: the changes made to the sudoku'''
#     isPoss, changes = waterfall(sudoku)
    #dms = get_domain(sudoku)
    changes = []
    arics = [i for i in arcs]
    while(len(arics) != 0):
        xi, xj = arics.pop(0)
        revised, changes = revise(sudoku, dms, xi, xj)
        if revised:
            if len(dms[xi[0]][xi[1]]) == 0:
                return False, changes
            neighbors = get_neighbors(xi)
            for n in neighbors:
                if n != xj:
                    arics.append((n, xi))
    return True, changes


def get_arcs(sudoku, dms):
    mappings = []
    for i in range(len(dms)):
        for j in range(len(dms)):
            if len(dms[i][j]) > 1:
                for k in range(len(dms)):
                    if len(dms[i][k]) > 1 and k != j:
                        mappings.append([(i ,j), (i, k)])
                for k in range(len(dms)):
                    if len(dms[k][j]) > 1 and k != i:
                        mappings.append([(i, j), (k, j)])
                for k in range((i // 3) * 3, ((i // 3) * 3)+3):
                    for l in range((j // 3) * 3, ((j // 3) * 3)+3):
                        if len(dms[k][l]) > 1  and (i!=k or j!=l):
                            mappings.append([(i, j), (k, l)])
    return mappings

def waterfall1(sudoku, dms, arcs):
    # Last Remaining Cell in a Box. If an element in the domain of a cell
    # is not found in domains of other cells in the same box, the domain of
    # the cell can be reduced to 1. We can fix the value for that location.
    '''The first waterfall method to apply
    input:  sudoku: the sudoku to apply the waterfall method on
            kwargs: the kwargs to be passed to the isPossible function
    output: isPoss: whether the sudoku is still possible to solve (i.e. not inconsistent))
            changes: the changes made to the sudoku'''
    changes = []
    to_keep = []
    for i in range(len(dms)):
        for j in range(len(dms)):
            if len(dms[i][j]) == 0:
                return False, changes
            if len(dms[i][j]) > 1:
                lst = []
                for k in range((i // 3) * 3, ((i // 3) * 3)+3):
                    for l in range((j // 3) * 3, ((j // 3) * 3)+3):
                        if i!=k or j!=l:
                            lst += dms[k][l]

                for value in dms[i][j]:
                    if value not in lst:
                        update_changes_for_position(sudoku, i, j, value)
                        break

    return True, changes


def solve_plain_backtracking(original_sudoku):
    '''Solve the sudoku using plain backtracking.'''
    sudoku = copy.deepcopy(original_sudoku)
    dms = get_domain(sudoku)
    arcs = get_arcs(sudoku, dms)
    ini_x, ini_y = 0, 0
    return solve_sudoku(sudoku, ini_x, ini_y, False, [], dms, arcs)

def solve_with_mrv(original_sudoku):
    '''Solve the sudoku using mrv heuristic.'''
    sudoku = copy.deepcopy(original_sudoku)
    dms = get_domain(sudoku)
    arcs = get_arcs(sudoku, dms)
    ini_x, ini_y = get_next_position_to_fill(sudoku, -1, -1, True, dms)
    return solve_sudoku(sudoku, ini_x, ini_y, True, [], dms, arcs)

def solve_with_ac3(original_sudoku):
    '''Solve the sudoku using mrv heuristic and ac3 waterfall method.'''
    sudoku = copy.deepcopy(original_sudoku)
    dms = get_domain(sudoku)
    arcs = get_arcs(sudoku, dms)
    all_waterfalls = [ac3_waterfall]
    ini_x, ini_y = get_next_position_to_fill(sudoku, -1, -1, True, dms)
    return solve_sudoku(sudoku, ini_x, ini_y, True, all_waterfalls, dms, arcs)

def solve_with_addition_of_waterfall1(original_sudoku):
    '''Solve the sudoku using mrv heuristic and waterfall1 waterfall method besides ac3.'''
    sudoku = copy.deepcopy(original_sudoku)
    dms = get_domain(sudoku)
    arcs = get_arcs(sudoku, dms)
    all_waterfalls = [ac3_waterfall, waterfall1]
    ini_x, ini_y = get_next_position_to_fill(sudoku, -1, -1, True, dms)
    return solve_sudoku(sudoku, ini_x, ini_y, True, all_waterfalls, dms, arcs)

def solve_one_puzzle(puzzle_path):

    sudoku = load_sudoku(puzzle_path)
    solved, solved_sudoku, backtracking_guesses = solve_plain_backtracking(sudoku)
    assert solved
    for r in solved_sudoku:
        print(r)
    solved, solved_sudoku, mrv_guesses = solve_with_mrv(sudoku)
    assert solved
    solved, solved_sudoku, ac3_guesses = solve_with_ac3(sudoku)
    assert solved
    solved, solved_sudoku, waterfall1_guesses = solve_with_addition_of_waterfall1(sudoku)
    assert solved
#     #Add more waterfall methods here if you want need to and return the number of guesses for each method
    return (backtracking_guesses, mrv_guesses, ac3_guesses, waterfall1_guesses)

def solve_all_sudoku():
    puzzles_folder = "puzzles"
    puzzles = os.listdir(puzzles_folder)
    puzzles.sort()
    for puzzle_file in puzzles:
        puzzle_path = os.path.join(puzzles_folder, puzzle_file)

        backtracking_guesses, mrv_guesses, ac3_guesses, waterfall1_guesses= solve_one_puzzle(puzzle_path)
        print("Puzzle: ", puzzle_file)
        print("backtracking guesses: ", backtracking_guesses)
        print("mrv guesses: ", mrv_guesses)
        print( "ac3 guesses: ", ac3_guesses)
        print("with waterfall1 guesses: ", waterfall1_guesses)


if __name__ == '__main__':
    solve_all_sudoku()
