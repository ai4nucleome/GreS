import configparser


class Config(object):
    def __init__(self, config_file):
        conf = configparser.ConfigParser()
        try:
            conf.read(config_file)
        except:
            print("loading config: %s failed" % (config_file))

        # Data settings
        self.fdim = conf.getint("Data_Setting", "fdim")
        
        # Graph parameters
        self.k = conf.getint("Model_Setup", "k")
        self.radius = conf.getint("Model_Setup", "radius")

        # Graph neighborhood sizes (backward-compatible fallbacks)
        self.fadj_k = conf.getint("Model_Setup", "fadj_k") if conf.has_option("Model_Setup", "fadj_k") else self.k
        self.sadj_k = conf.getint("Model_Setup", "sadj_k") if conf.has_option("Model_Setup", "sadj_k") else self.radius

        # Training parameters
        self.seed = conf.getint("Model_Setup", "seed")
        self.lr = conf.getfloat("Model_Setup", "lr")
        self.weight_decay = conf.getfloat("Model_Setup", "weight_decay")
        self.epochs = conf.getint("Model_Setup", "epochs")
        
        # Model architecture
        self.nhid1 = conf.getint("Model_Setup", "nhid1")
        self.nhid2 = conf.getint("Model_Setup", "nhid2")
        self.dropout = conf.getfloat("Model_Setup", "dropout")
        self.llm_modulation_ratio = conf.getfloat("Model_Setup", "llm_modulation_ratio")

        # Loss weights
        self.alpha = conf.getfloat("Model_Setup", "alpha")
        self.beta = conf.getfloat("Model_Setup", "beta")
        self.gamma = conf.getfloat("Model_Setup", "gamma")
        
        # Runtime options
        self.no_cuda = conf.getboolean("Model_Setup", "no_cuda")
        self.no_seed = conf.getboolean("Model_Setup", "no_seed")
