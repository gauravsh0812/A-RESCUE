# NOTE: Kindly change the paths as per your system.

import os
from subprocess import call

def main():
	
	#HETEROGENEOUS BENCHMARKS - same as homogeneous combinations
	benchmarks = ["360buy.rcS", "adobe.rcS", "baidumap.rcS", "bbench.rcS", "frozenbubble.rcS", "k9mail.rcS", "kingsoftoffice.rcS", "mxplayer.rcS", "netease.rcS", "sinaweibo.rcS", "ttpod.rcS"]

	retention_d = ["75M","100M", "400M", "1G", "10G", "100G"] #M is micro, G is millie, T is seconds      d-cache
	retention_i = ["1G", "50G", "100G", "500G", "1T"] #M is micro, G is millie, T is seconds     i-cache

	# root directory --> CHANGE
	dir_0= ("/home/gauravs")
	# join to gem5 folder 
	dir_1 = os.join.path(dir_0, "---")

	# make results folder --> "ResultsForRerun"
	call(['mkdir', ResultsForRerun])
	dir_2 = os.join.path(dir_1, "ResultsForRerun")

	for i in benchmarks:
	    # dir for results of simulation of each benchmark
	    dir_3 = call(['mkdir', {}]).format(i)

	    for dret in retention_d:
		for iret in retention_i:

		    # path to asimbench directory --> CHANGE
		    script = ("--script=/home/gauravs/GEM5_Tosi/asimbench/boot/{}").format(i)

		    # running the simulation --> CHANGE
		    cmd = ("./build/ARM/gem5.opt --stats-file={}_75M_100G.txt configs/example/fs.py --caches --l1i_size=32kB --l1i_assoc=4 --l1d_size=32kB --l1d_assoc=4 \
		    --l2_size=128KB  --l2_assoc=8 --cacheline_size=64 --cpu-clock=2GHz --mem-size=256MB --cpu-type=arm_detailed --l1d_ret={} --l1i_ret={} --maxinsts=1000000000 \
		    --script={} --disk-image=/home/gauravs/asimbench/disks/ARMv7a-ICS-Android.SMP.Asimbench-v3.img \
		    --machine-type=RealView_PBX --os-type=android-ics -F 5000000000").format(i, dret,iret,script)

		    os.system(cmd)

		    # path to m5out --> CHANGE
		    m5out= "/home/gauravs/GEM5_Tosi/gem5_ret_aware/m5out/"
		    os.chdir(m5out)
		    src = ('stats.txt')
		    dst = ('{}_iWRlat=3_iret={}_dWRlat=2_dret={}.txt').format(i,iret, dret)
		    os.rename(src,dst)
		    call(["mv", dst ,dir_3])

		    # Path to gem5 dir 
		    gem5_ret_aware= "/home/gauravs/GEM5_Tosi/gem5_ret_aware/"
		    os.chdir(gem5_ret_aware)


if __name__=='__main__':
	main()

