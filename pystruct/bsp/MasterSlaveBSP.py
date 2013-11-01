from BSP import BSP
import traceback

import numpy as np
import io

class MasterSlaveBSP(BSP):
    def setup(self, peer):
        self.master_index = peer.config.get("master.index")
        train = peer.config.get("mode") == "train"

        if peer.getPeerIndex() == self.master_index:
            self.impl = MasterBSP()
        else:
            if train:
              self.impl = SlaveBSPTrain(self.master_index)
            else:
              self.impl = SlaveBSPTest(self.master_index)

        try:
            self.impl.setup(peer)
        except:
            peer.log("".join(traceback.format_exc().splitlines()))

    def bsp(self, peer):
        try:
            self.impl.bsp(peer)
        except:
            peer.log("".join(traceback.format_exc().splitlines()))

    def cleanup(self, peer):
        try:
            self.impl.cleanup(peer)
        except:
            peer.log("".join(traceback.format_exc().splitlines()))
            
            
class SlaveBSP:
    def __init__(self, master_id):
        self.master_id = master_id

    def setup(self, peer):
        self.X = []
        self.Y = []
        self.Y_areas = []
        self.Psi_gt = []
        self.imgnums = []
        
        module = peer.config.get("model.factory.module")
        func = peer.config.get("model.factory.function")
        self.model = getattr(__import__(module), func)()
        
        while True:
            line = peer.readNext()

            if not line:
                break
            #peer.log("%s, %s" %(line[0], line[1]))
            
            un_str, pw_str, lab_str, *tmp = line[1].split(";")
            
            un_feat = np.genfromtxt(io.BytesIO(un_str.replace(",","\n").encode()))
            #peer.log("Unary shape: %s" % str(un_feat.shape))
            pw_feat = np.genfromtxt(io.BytesIO(pw_str.replace(",","\n").encode()))
            #peer.log("Pairwise shape: %s" % str(pw_feat.shape))
            lab = np.genfromtxt(io.BytesIO(lab_str.replace(",","\n").encode()))
            #peer.log("Labels shape: %s" % str(lab.shape))
            
            img_num = un_feat[0,0]
            assert(np.all(un_feat[:,0] == img_num))
            assert(np.all(pw_feat[:,0] == img_num))
            assert(np.all(lab[:,0] == img_num))
            
            self.imgnums.append(img_num)
            self.X.append((un_feat[:,2:],pw_feat[:,1:3].astype(int)-1,pw_feat[:,3:]))
            self.Y.append(lab[:,2].astype(int)-1)
            
            if peer.config.get("mode") != "train":   # this should be done in a subclass
                self.Y_areas.append(lab[:,2].astype(int)-1)  # For MSRC format, use lab[:,3:] for areas

            if peer.config.get("mode") == "train":   # this should be done in a subclass
                self.Psi_gt.append(self.model.psi(self.X[-1], self.Y[-1], 
                    self.Y[-1] if getattr(self.model, 'rescale_C', False) else None))
            peer.log("Object %d processed!" % img_num)
            
        self.sum_psi_gt = sum(self.Psi_gt) 
                

    def cleanup(self, peer):
        pass

        

class SlaveBSPTrain(SlaveBSP):
    def bsp(self, peer):
        while self.superstep(peer):
            pass
            
    def superstep(self, peer):
        peer.sync()
        msg = peer.getCurrentMessage()
        if not msg:
            return False

        w = np.array([float(elem) for elem in msg.split()])
        
        Y_hat = self.model.batch_loss_augmented_inference(self.X, self.Y, w, relaxed=False)
        
        Dpsi = [psi_gt - 
                self.model.psi(x, y_hat, 
                    y if getattr(self.model, 'rescale_C', False) else None)
            for x, y_hat, y, psi_gt in zip(self.X, Y_hat, self.Y, self.Psi_gt)]
        sum_dpsi = sum(Dpsi)

        Loss = [self.model.loss(y, y_hat) for y, y_hat in zip(self.Y, Y_hat)]
        sum_loss = sum(Loss)
        
        # there is some redundancy here, but it moves some computation to slaves
        peer.send(peer.getPeerNameForIndex(self.master_id), 
                ",".join(" ".join(str(i) for i in y_hat) for y_hat in Y_hat) + ";" + 
                ",".join(" ".join(str(i) for i in dpsi) for dpsi in Dpsi) + ";" + 
                " ".join(str(i) for i in sum_dpsi) + ";" + 
                " ".join(str(i) for i in Loss) + ";" + str(sum_loss))
        peer.sync()
        
        return True
            
            
class SlaveBSPTest(SlaveBSP):
    def bsp(self, peer):
        peer.sync()
        msg = peer.getCurrentMessage()
        assert(msg)
        
        w = np.array([float(elem) for elem in msg.split()])
        
        Y_hat = self.model.batch_inference(self.X, w, relaxed=False)
        
        peer.send(peer.getPeerNameForIndex(self.master_id), 
                ",".join(" ".join(str(i) for i in y_hat) for y_hat in Y_hat) + ";" + 
                #",".join(" ".join(str(i) for i in y_areas.reshape(1, -1)) for y_areas in Y_areas))
                ",".join(" ".join(str(i) for i in y_areas) for y_areas in self.Y_areas))
        peer.sync()
        
        
        
class MasterBSP:
    def setup(self, peer):
        pass
        #if self.squire is not None:
        #    self.squire.setup(peer)

    def bsp(self, peer):
        module = peer.config.get("entry.point.module")
        func = peer.config.get("entry.point.function")
        getattr(__import__(module), func)(peer)
        
        #peer.write(result, "")
        
    def cleanup(self, peer):
        pass
