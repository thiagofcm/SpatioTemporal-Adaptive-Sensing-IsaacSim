"""Minimal Vulkan probe: create instance, enumerate physical devices, print names."""
import ctypes, sys

lib = ctypes.CDLL(sys.argv[1] if len(sys.argv) > 1 else "libvulkan.so.1")

class AppInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("pApplicationName", ctypes.c_char_p),
                ("applicationVersion", ctypes.c_uint32), ("pEngineName", ctypes.c_char_p),
                ("engineVersion", ctypes.c_uint32), ("apiVersion", ctypes.c_uint32)]

class InstInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("pApplicationInfo", ctypes.POINTER(AppInfo)), ("enabledLayerCount", ctypes.c_uint32),
                ("ppEnabledLayerNames", ctypes.c_void_p), ("enabledExtensionCount", ctypes.c_uint32),
                ("ppEnabledExtensionNames", ctypes.c_void_p)]

class Props(ctypes.Structure):
    _fields_ = [("apiVersion", ctypes.c_uint32), ("driverVersion", ctypes.c_uint32), ("vendorID", ctypes.c_uint32),
                ("deviceID", ctypes.c_uint32), ("deviceType", ctypes.c_int), ("deviceName", ctypes.c_char * 256),
                ("pad", ctypes.c_byte * 2048)]

app = AppInfo(0, None, b"probe", 1, b"probe", 1, (1 << 22) | (3 << 12))
info = InstInfo(1, None, 0, ctypes.pointer(app), 0, None, 0, None)
inst = ctypes.c_void_p()
r = lib.vkCreateInstance(ctypes.byref(info), None, ctypes.byref(inst))
print("vkCreateInstance ->", r)
if r != 0:
    sys.exit(1)
n = ctypes.c_uint32(0)
r = lib.vkEnumeratePhysicalDevices(inst, ctypes.byref(n), None)
print("vkEnumeratePhysicalDevices ->", r, "count", n.value)
devs = (ctypes.c_void_p * n.value)()
lib.vkEnumeratePhysicalDevices(inst, ctypes.byref(n), devs)
for d in devs:
    p = Props()
    lib.vkGetPhysicalDeviceProperties(ctypes.c_void_p(d), ctypes.byref(p))
    print(f"  {p.deviceName.decode()}  type={p.deviceType} api={p.apiVersion>>22}.{(p.apiVersion>>12)&0x3ff}")
