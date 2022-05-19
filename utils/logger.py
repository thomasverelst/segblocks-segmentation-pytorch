import logging
import torch
import pprint
import sys, os
import json
import numpy as np

class AdvLoggerLite(logging.Logger):
    def __init__(self, name, level = logging.INFO):
        super().__init__(name, level)
        self._is_init = False
        self.step = 0

    def init(self, save_path, interval):
        """
        Initialize the logger.
        """
        self.save_path = save_path
        self.interval = interval
        self.use_aim = False

        FORMAT = '%(asctime)s - %(message)s'
            
        handlers = [
            logging.StreamHandler(stream=sys.stdout)
        ]

        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            handlers.append(logging.FileHandler(os.path.join(self.save_path, "run.log")))
        
        logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S', handlers=handlers)
        self.loggers = {}
        self._is_init = True


    ## LOGGERS
    
    @staticmethod
    def cleanval(val):
        ''' cleans the given value '''
        if callable(val):
            return val()
        if isinstance(val, dict):
            # dict to str
            try:
                val = '\n'+json.dumps((val), indent=4, sort_keys=True)
            except:
                val = '\n'+pprint.pformat(val)
        
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val)
                
        if torch.is_tensor(val):
            if val.dim() == 0:
                val = val.item()
            elif val.numel() == 1:
                val = val.item()
        
        return val

    def get_step(self):
        return self.step 
    
    def log_float_interval(self, name, val, log_aim=True, use_interval=True):
        """
        Log a float value.
        """
        if self.istick() or not use_interval:
            val = float(self.cleanval(val))
            step = self.get_step()
            self._add(name, val, step)
            
    def log_float(self, name, val, log_aim=True):
        """
        Log a float at every call, does not use interval
        """
        self.log_float_interval(name, val, log_aim, use_interval=False)
            
    def log_text_interval(self, name, val, log_aim=True, use_interval=True, is_param=False):
        """
        Log a text string 
        """
        if self.istick() or not use_interval:
            val = str(self.cleanval(val))
            step = self.get_step()
            self._add(name, val, step)
            
    def log_text(self, name, val, log_aim=True, is_param=False):
        self.log_text(name, val, log_aim, use_interval=False, is_param=is_param)

    def _add(self, name, val, step=None):
        self.loggers[name] = (step, val)

     ## PRINT FUNCTIONS
    def whitespace(self, n=1):
        for _ in range(n):
            self.info('')

    def header(self, text='', width=120, char1='#', char2='#', n_front=1):
        self.whitespace(n_front)
        sidelength = (width - len(text) - 4) // 2
        s = str(char1) + str(char2)*sidelength + ' '+ str(text) + ' ' + str(char2)*(sidelength + len(text) % 2) + str(char1)
        self.info(s)
    
    def subheader(self, text='', width=120, char1='#', char2='-', n_front=0):
        self.header(text, width, char1, char2, n_front)
        
    def _do_print(self, name, val):
        if isinstance(val, float):
            self.info('$ {:15}: {:10f}'.format(name, val))
        else:
            self.info('$ {:15}: {}'.format(name, val))

    ## TICK and OUT
    def tick(self):
        if self.step > 0 and self.step % self.interval == 0:
            self.out()
        self.step += 1

    def istick(self, interval=None, interval_factor=None):
        interval = self.interval if interval is None else interval
        interval = interval if interval_factor is None else interval*interval_factor
        return interval == 0 or self.step % interval == 0

    def reset(self):
        self.out()
        self.loggers = {}

    def out(self):
        assert self._is_init
        self.info(f'$$$ Step {self.step} [interval {self.interval}]')
        for name in sorted(self.loggers):
            step, val = self.loggers[name]
            if val is not None:
                self._do_print(name, val)
        self.info(' ')
        self.loggers = {}


logging.setLoggerClass(AdvLoggerLite)
logger = logging.getLogger('default') 