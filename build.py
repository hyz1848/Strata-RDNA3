import sys,time
sys.path.insert(0,"/opt/Strata-022/Strata-0.2.2")
import importlib.util
spec=importlib.util.spec_from_file_location("s022","/opt/Strata-022/Strata-0.2.2/setup.py")
m=importlib.util.module_from_spec(spec); sys.modules["s022"]=m; spec.loader.exec_module(m)
t=time.time()
eng=m.build_engine_hip({"archs":["gfx1100"],"arch":"gfx1100","index":0,"count":2}, m.ROOT/"third_party"/"llama.cpp","cpu")
print("BUILD_OK",eng,(eng/m.EXE).stat().st_size,f"{time.time()-t:.0f}s")
