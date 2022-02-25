# Catchement class

from utils import utils, myFile

class catchment_param:
    def __init__(self) -> None:
        self.CM = int(0)
        self.inflow_file = myFile() # myFile type from utils!


class catchment:
    def __init__(self, pCM):
        cModel = pCM.CM
        self.inflow = utils.loadVector(pCM.inflow_file.filename, pCM.inflow_file.row)

        if cModel == 0:
            self.inflow = utils.loadVector(pCM.inflow_file.filename, pCM.inflow_file.row)

    def getInflow(self, pt):

        q = float(self.inflow[pt])
        return q