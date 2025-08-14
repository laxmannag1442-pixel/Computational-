class laxmanlibrary:
  # matrices multiplication and vector dot product
  def read_matrix(self,filename):
      with open(filename,'r') as file:
        matrix = []
        for line in file:
        # Split the line into numbers and convert to int/float
          row =[float(num)for num in line.strip().split()]
          matrix.append(row)
        return matrix
        
  def lcg(self,seed):
    a = 1103515245 
    c = 12345
    m = 32768
    list =[seed]
    for _ in range(9999):
      seed = (a*seed +c)%m
      list.append(seed)
    return list
      
    
    
