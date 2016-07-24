.onUnload <- function (libpath) {
  library.dynam.unload("KDRcpp", libpath)
}