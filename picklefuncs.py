def stringFunc( matrix, row, mul ):
    new_row = matrix[row,:] * mul
    return( '%d, %d: %s' % (row, mul, str( new_row )) )

def anotherFunc( row, matrix, action, num ):
    if action == 'add':
        new_row = matrix[row,:]
        for i in range( len( new_row ) ):
            new_row[i] += num
    elif action == 'multiply':
        new_row = matrix[row,:] * num
    return( '%s %d to row %d: %s' % (action, num, row, str( new_row )) )

def sum_row( matrix, row ):
    return( sum( matrix[row,:] ) )