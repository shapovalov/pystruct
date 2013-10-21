from BSP import BSP
import traceback

import numpy as np
import io

class MasterSlaveBSP(BSP):
    def setup(self, peer):
        self.master_index = peer.config.get("master.index")

        if peer.getPeerIndex() == self.master_index:
            self.impl = MasterBSP()
        else:
            self.impl = SlaveBSP(self.master_index)

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
            pw_feat = np.genfromtxt(io.BytesIO(pw_str.replace(",","\n").encode()))
            lab_feat = np.genfromtxt(io.BytesIO(lab_str.replace(",","\n").encode()))
            
            img_num = un_feat[0,0]
            assert(np.any(un_feat[:,0] == img_num))
            assert(np.any(pw_feat[:,0] == img_num))
            assert(np.any(lab_feat[:,0] == img_num))
            
            self.imgnums.append(img_num)
            self.X.append((un_feat[:,2:],pw_feat[:,1:3],pw_feat[:,3:]))
            self.Y.append(lab_feat[:,2])
            peer.log("Object %d processed!" % img_num)
                

    def bsp(self, peer):
        while self.superstep(peer):
            pass

    def cleanup(self, peer):
        pass

    def superstep(self, peer):
        peer.sync()
        msg = peer.getCurrentMessage()
        if not msg:
            return False

        w = np.array([int(elem) for elem in msg.split()])
        
        Y_hat = self.model.batch_loss_augmented_inference(X, Y, w, relaxed=False)
        sum_psi = self.model.batch_psi(X, Y_hat, 
            Y if getattr(self.model, 'rescale_C', False) else None)

        sum_loss = np.sum(self.model.batch_loss(Y, Y_hat))
        
        peer.send(peer.getPeerNameForIndex(self.master_id), 
                ",".join(" ".join(str(i) for i in y_hat) for y_hat in Y_hat) + ";" + 
                " ".join(str(i) for i in sum_psi) + ";" + str(sum_loss))
        peer.sync()
        
        return True
        
        
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
        if self.squire is not None:
            self.squire.cleanup(peer)
