import subprocess
import numpy as np

def qflll(mat):
    """
    This is definitely a hack but it works. 
    """
    # create a file in /tmp containing the qfill script
    M = mat.shape[0]
    p = [',',';']
    elts = ''.join([str(mat.item(i))+p[0 if (i+1)%M else 1] for i in xrange(M**2)])[:-1]
    script = """print(qflll([%s]))\nquit"""%(elts)
    f = open('/tmp/abel.gp','w')
    f.write(script)
    f.close()

    # run the command and get the string output
    out = subprocess.check_output(['gp','-q','/tmp/abel.gp'])
    
    # build matrix and return
    mat  = np.matrix(out)
    return mat
