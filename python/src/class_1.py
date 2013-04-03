class IStudent:
    def __init__(self):
        self.name=""
        self.sex=""
        self.ID=""
        self.type=""
        print("this is IStudent")
    @property
    def Name(self):
        return self.name
    @Name.setter
    def Name(self,value):
        self.name=value
        return;
    @property
    def Sex(self):
        return self.sex
    @Sex.setter
    def Sex(self,value):
        self.sex=value
        return
    @property
    def Type(self):
        return self.type
    @Type.setter
    def Type(self,value):
        self.type=value
        return
class Master(IStudent):
    def __init__(self):
        super(Master,self).__init__()
        self.type="Master"
        print("this is %s" %(self.type))
a1=Master()
a1.Name="nihao"
print(a1.Name)


        