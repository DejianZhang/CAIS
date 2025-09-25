import geatpy as ea  
from MyProblem import MyProblem


if __name__ == '__main__':

    problem = MyProblem(PoolType='Process')  

    algorithm = ea.moea_NSGA2_templet(
        problem,
        ea.Population(Encoding='RI', NIND=20),
        MAXGEN=150,  
        logTras=1,  
        trappedValue=1e-6,  
        maxTrappedCount=100  
    )
    algorithm.mutOper.Pm = 0.4  
    algorithm.recOper.XOVR = 0.6  
       
    res = ea.optimize(algorithm,
                      verbose=False,
                      drawing=1,
                      outputMsg=True,
                      drawLog=False,
                      saveFlag=True)
    print(res)
