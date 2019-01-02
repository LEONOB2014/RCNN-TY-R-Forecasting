import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
from args_tools import *

def output_figure():
    print("*" * 20)
    print("*{:^18s}*".format('Figure maker'))
    print("*" * 20)

    # Taipei
    study_area = args.study_area
    # set lat and lon Taipei
    lat_l = args.lat_l
    lat_h = args.lat_h
    lon_l = args.lon_l
    lon_h = args.lon_h
    # Set path
    wrangled_files_folder = args.wrangled_files_folder+"_"+study_area
    wrangled_figs_folder = args.wrangled_figs_folder+"_"+study_area

    TW_map_file = args.TW_map_file
    # dropbox_fig_folder = args.dropbox_folder+"/01_TY_database/03_wrangled_fig_"+study_area

    ty_list = pd.read_excel(args.ty_list)
    sta_list = pd.read_excel(args.sta_list, index_col="NO")

    # set specific color for radar, qpe, and qpf data
    levels_qp = [-5,0,10,20,35,50,80,120,160,200]
    c_qp = ('#FFFFFF','#D2D2FF','#AAAAFF','#8282FF','#6A6AFF','#4242FF','#1A1AFF','#000090','#000040','#000030')
    levels_rad = [-5,0,10,20,30,40,50,60,70]
    c_rad = ('#FFFFFF','#FFD8D8','#FFB8B8','#FF9090','#FF6060','#FF2020','#CC0000','#A00000')

    for i in sorted(os.listdir(wrangled_files_folder)):
        if i != "RAD":
            continue
        else:
            data_path = wrangled_files_folder+'/'+i
            fig_path = wrangled_figs_folder+'/'+i
            createfolder(fig_path)
            print("—" * 20)
            print("|{:^18s}|".format(i))
            print("—" * 20)
            tmp=0
            for j in sorted(os.listdir(data_path)):
                if j[:-17] != tmp :
                    tmp = j[:-17]
                    print("|{:^18s}|".format(tmp))
                file_in = data_path+"/"+j
                data = np.load(file_in)
                x = np.linspace(lon_l,lon_h,len(data))
                y = np.linspace(lat_l,lat_h,len(data))
                plt.figure(figsize=(6,5))
                ax = plt.gca()
                m = Basemap(projection='cyl',resolution='h', llcrnrlat=lat_l, urcrnrlat=lat_h, llcrnrlon=lon_l, urcrnrlon=lon_h)
                m.readshapefile(TW_map_file, name='Taiwan', linewidth=0.25, drawbounds=True, color='k')
                X, Y = np.meshgrid(x,y)

                if i == "QPE" or i == "QPF":
                    cp = m.contourf(X,Y,data,levels_qp,colors=c_qp,alpha=0.95)
                    cbar.set_label('Rainfall (mm)',fontsize=10)
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.1)
                    cbar = plt.colorbar(cp,cax=cax)
                    plt.plot(sta_list["Longitude"].iloc[:9].values, sta_list["Latitude"].iloc[:9].values, marker='^', mfc='r', mec='k', linestyle='None', markeredgewidth = 0.25, markersize=4, label="Man station")
                    plt.plot(sta_list["Longitude"].iloc[9:].values, sta_list["Latitude"].iloc[9:].values, marker='.', mfc='k', mec='k', linestyle='None', markersize=1.5, label="Auto station")
                    plt.legend(fontsize=7,loc=1)
                else:
                    cp = m.contourf(X,Y,data,levels_rad,colors=c_rad,alpha=0.95)
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.1)
                    cbar = plt.colorbar(cp,cax=cax)
                    cbar.set_label('Radar reflection (dbz)',fontsize=9)

                cbar.ax.tick_params(labelsize=8)
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                title = j[:-17]
                ax.set_title(title,fontsize = 12)
                text_time = ""+j[-16:-12]+'/'+j[-12:-10]+"/"+j[-10:-8]+" "+j[-8:-6]+':'+j[-6:-4]
                ax.text(122.12,26.3,text_time,fontsize = 7)
                ax.set_xlabel(r"longitude$(^o)$",fontsize = 10)
                ax.set_ylabel(r"latitude$(^o)$",fontsize = 10)
                ax.set_xticks(np.linspace(args.lon_l,args.lon_h,6))
                ax.set_yticks(np.linspace(args.lat_l,args.lat_h,6))
                ax.tick_params(labelsize=8)
                figname = fig_path+'/'+j[:-4]+'.png'
                plt.savefig(figname,dpi=300,bbox_inches='tight')
                plt.close()

            print("—" * 20)

if __name__ == "__main__":
    output_figure()
