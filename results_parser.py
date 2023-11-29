# import required module
import os
import re
from decimal import *

# 7-bit C1 ANSI sequences
ansi_escape = re.compile(r'''
    \x1B  # ESC
    (?:   # 7-bit C1 Fe (except CSI)
        [@-Z\\-_]
    |     # or [ for CSI, followed by a control sequence
        \[
        [0-?]*  # Parameter bytes
        [ -/]*  # Intermediate bytes
        [@-~]   # Final byte
    )
''', re.VERBOSE)
outlines = []

bound_list = []

filename_full = 'output2parse.txt'

with open(filename_full, 'r', encoding='ASCII') as f:
    verify_result = ''
    num_pipes = ''
    num_branches = ''
    total_time = ''
    dnn_time = ''
    left_x1_bound = ''
    right_x1_bound = ''
    left_x2_bound = ''
    right_x2_bound = ''
    left_x3_bound = ''
    right_x3_bound = ''
    lines = f.readlines()
    output_empty = False
    total_times = []
    if lines == []:
        print("no results found in file")
    else:
        lines = [ansi_escape.sub('', line) for line in lines]
        
        lines = [line.strip() for line in lines if line != '' and line !='\n']

        
        
        i = 0
        while i < len(lines):
            curline = lines[i].strip()
            word_list = curline.split()
            # make sure it is completed and update verification result
            if 'Computation completed' in lines[i]:
                if 'SAFE' in lines[i+1]:
                    verify_result = 'SAFE'
                    num_pipes = word_list[2]
                    total_time = lines[i+2].split()[-2]
                    num_branches = lines[i+3].split()[-1]
                    dnn_time = lines[i+4].split()[-1]                    
                    left_x1_bound = lines[i+6].split()[0][1:-1]
                    right_x1_bound = lines[i+6].split()[1][0:-1]
                    bound_list.append([Decimal(left_x1_bound), Decimal(right_x1_bound)])
                    outline_list = [left_x1_bound, right_x1_bound, verify_result, total_time, dnn_time, num_branches, num_pipes]
                    outline_str = ', '.join(outline_list)
                    outlines.append(outline_str)
                    total_times.append(float(total_time))


                
            i = i+1
           # handle the cases where the intervals do not cover all space
# sanity check: 

#sort the tuples
sorted_bound_list = sorted(bound_list, key= lambda x: x[0])

cur_range = sorted_bound_list[0]
covered_ranges = []
uncovered_ranges = []
for i in range(1, len(sorted_bound_list)):
    next_bound = sorted_bound_list[i]
    # if the next candidate starts within our current range or abutts it
    if next_bound[0] <= cur_range[1]: 
        # that means we can absorb it into current range
        # but we need to decide to expand range or not
        if next_bound[1] > cur_range[1]:
            cur_range[1] = next_bound[1]
    # this means we cannot abosorb into current range, so we must split and save the covered regions and uncovered region, update cur_range to new bounds
    else:
        covered_ranges.append(cur_range)
        uncovered_ranges.append([cur_range[1], next_bound[0]])
        cur_range = next_bound
covered_ranges.append(cur_range)  
covered_ranges = [str(l) + ', ' +str(r) for [l,r] in covered_ranges]  
uncovered_ranges = [str(l) + ', ' +str(r) for [l,r] in uncovered_ranges] 


# now join the individual lines together
print('Total Completed: ', len(total_times))
print('AVERAGE TIME: ', sum(total_times)/len(total_times))
print('Number of runs under 2 hours: ', len([time for time in total_times if time < 7200]))
print('Number of runs btwn 2 and 5 hours: ', len([time for time in total_times if time >= 7200 and time <18000]))
print('Number of runs btwn 5 and 10 hours: ', len([time for time in total_times if time >= 18000 and time < 36000]))
print('Number of runs above 10 hours: ', len([time for time in total_times if time >= 36000])) 
header_line = 'X1 lower, X1 upper, Result, Total Time, DNN Time, Num Branches, Num Flowpipes\n'
outstring = header_line + '\n'.join(outlines) + '\nCovered Ranges:\n' + '\n'.join(covered_ranges) + '\nUncovered Ranges:\n' + '\n'.join(uncovered_ranges)
outfile_name = 'RESULTS_SUMMARY.txt'
g = open(outfile_name, 'w')
g.write(outstring)
g.close()
f.close()
