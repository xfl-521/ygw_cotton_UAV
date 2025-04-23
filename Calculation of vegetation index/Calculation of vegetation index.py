import os  
from osgeo import gdal  
import numpy as np  
  
def find_band_file(folder, band_name):  
    """在指定文件夹中查找特定波段的TIFF文件"""  
    for file in os.listdir(folder):  
        if band_name.lower() in file.lower() and file.lower().endswith('.tif'):  
            return os.path.join(folder, file)  
    return None  
  
def calculate_ndvi(red_band, nir_band):  
    """归一化差分植被指数（Normalized Difference Vegetation Index, NDVI）"""  
    np.seterr(divide='ignore', invalid='ignore')
    # NDVI公式: NDVI = (NIR - Red) / (NIR + Red)
    ndvi = (nir_band.astype(float) - red_band.astype(float)) / (nir_band.astype(float) + red_band.astype(float))  
    ndvi[np.isnan(ndvi)] = 0  # 将NaN值替换为0
    mask = np.isinf(ndvi)
    max = np.max(ndvi[~mask])
    ndvi[mask] = max
    return ndvi     
  
def calculate_gndvi(green_band, nir_band):  
    """绿色归一化差分植被指数（Green Normalized Difference Vegetation, GNDVI）"""  
    np.seterr(divide='ignore', invalid='ignore') 
    # GNDVI公式: GNDVI = (NIR - Green) / (NIR + Green)
    gndvi = (nir_band.astype(float) - green_band.astype(float)) / (nir_band.astype(float) + green_band.astype(float))  
    gndvi[np.isnan(gndvi)] = 0  # 将NaN值替换为0  
    mask = np.isinf(gndvi)
    max = np.max(gndvi[~mask])
    gndvi[mask] = max
    return gndvi

def calculate_rendvi(red_band, rededge_band):  
    """红边归一化差分植被指数（Rededge Normalized Difference Vegetation Index, RENDVI）"""  
    np.seterr(divide='ignore', invalid='ignore')
    # RENDVI公式: RENDVI = (Rededge - Red) / (Rededge + Red)
    rendvi = (rededge_band.astype(float) - red_band.astype(float)) / (rededge_band.astype(float) + red_band.astype(float))
    rendvi[np.isnan(rendvi)] = 0  # 将NaN值替换为0
    mask = np.isinf(rendvi)
    max = np.max(rendvi[~mask])
    rendvi[mask] = max
    return rendvi

def calculate_cig(green_band, nir_band):  
    """叶绿素绿指数（Chlorophyll index green, CIG）"""  
    np.seterr(divide='ignore', invalid='ignore')
    # CIG公式: CIG = NIR / Green - 1
    cig = nir_band.astype(float) / green_band.astype(float) - 1  
    cig[np.isnan(cig)] = 0  # 将NaN值替换为0
    mask = np.isinf(cig)
    max = np.max(cig[~mask])
    cig[mask] = max
    return cig    

def calculate_ngr(green_band, nir_band):  
    """近红外-绿比值植被指数(NIR-Green ratio, NGR)"""  
    np.seterr(divide='ignore', invalid='ignore')
    # NGR公式: NGR = NIR / Green
    ngr = nir_band.astype(float) / green_band.astype(float)   
    ngr[np.isnan(ngr)] = 0  # 将NaN值替换为0
    mask = np.isinf(ngr)
    max = np.max(ngr[~mask])
    ngr[mask] = max
    return ngr

def calculate_nrr(nir_band, red_band):  
    """近红外-红比值植被指数(NIR-Red ratio, NRR)"""  
    np.seterr(divide='ignore', invalid='ignore')
    # NRR公式: NRR = NIR / Red
    nrr = nir_band.astype(float) / red_band.astype(float)   
    nrr[np.isnan(nrr)] = 0  # 将NaN值替换为0
    mask = np.isinf(nrr)
    max = np.max(nrr[~mask])
    nrr[mask] = max
    return nrr

def calculate_rri(rededge_band, red_band):  
    """红边-红比值指数(Rededge-red ratio index, RRI)"""  
    np.seterr(divide='ignore', invalid='ignore') 
    # NRERVI公式: NRERVI = Rededge / Red
    rri = rededge_band.astype(float) / red_band.astype(float)   
    rri[np.isnan(rri)] = 0  # 将NaN值替换为0
    mask = np.isinf(rri)
    max = np.max(rri[~mask])
    rri[mask] = max
    return rri

def calculate_dvi(red_band, nir_band):  
    """差分植被指数（Difference Vegetation Index, DVI）"""  
    np.seterr(divide='ignore', invalid='ignore') 
    # DVI公式: DVI = NIR - Red
    dvi = nir_band.astype(float) - red_band.astype(float)   
    dvi[np.isnan(dvi)] = 0  # 将NaN值替换为0 
    mask = np.isinf(dvi)
    max = np.max(dvi[~mask])
    dvi[mask] = max
    return dvi

def calculate_savi(red_band, nir_band):  
    """优化土壤调整植被指数（Soil-Adjusted Vegetation Index, SAVI）"""  
    np.seterr(divide='ignore', invalid='ignore')  
    # SAVI公式: SAVI = [1.5 * (NIR - Red)] / (NIR + Red + 0.5) 
    savi = (1.5 * (nir_band.astype(float) - red_band.astype(float))) / (nir_band.astype(float) + red_band.astype(float) + 0.5)  
    savi[np.isnan(savi)] = 0  # 将NaN值替换为0  
    mask = np.isinf(savi)
    max = np.max(savi[~mask])
    savi[mask] = max
    return savi

def calculate_osavi(red_band, nir_band):  
    """优化土壤调整植被指数（Optimized Soil-Adjusted Vegetation Index, OSAVI）"""  
    np.seterr(divide='ignore', invalid='ignore')  
    # OSAVI公式: OSAVI = [1.16*(NIR - Red)] / (NIR + Red + 0.16) 
    osavi = (1.16 * (nir_band.astype(float) - red_band.astype(float))) / (nir_band.astype(float) + red_band.astype(float) + 0.16)  
    osavi[np.isnan(osavi)] = 0  # 将NaN值替换为0  
    mask = np.isinf(osavi)
    max = np.max(osavi[~mask])
    osavi[mask] = max
    return osavi  
  
def calculate_mcari(red_band, green_band, nir_band):  
    """修正的叶绿素吸收比值指数（Modified Chlorophyll Absorption Ratio Index, MCARI）"""  
    np.seterr(divide='ignore', invalid='ignore')  
    # MCARI公式: MCARI = [(NIR - Red) - 0.2 * (NIR - Green)] * (NIR / Red)  
    mcari = ((nir_band.astype(float) - red_band.astype(float)) - 0.2 * (nir_band.astype(float) - green_band.astype(float))) * (nir_band.astype(float) / red_band.astype(float))
    mcari[np.isnan(mcari)] = 0  # 将NaN值替换为0  
    mask = np.isinf(mcari)
    max = np.max(mcari[~mask])
    mcari[mask] = max
    return mcari

def calculate_tcari(red_band, green_band, rededge_band):  
    """转化叶绿素吸收反射率（Transformed Chlorophyll Absorption Reflectance Index, TCARI）"""  
    np.seterr(divide='ignore', invalid='ignore')  
    # TCARI公式: TCARI = 3 * [(Rededge - Red) - 0.2 * (Rededge - Green) * (Rededge / Red)]   
    tcari = 3 * ((rededge_band.astype(float) - red_band.astype(float)) - 0.2 * (rededge_band.astype(float) - green_band.astype(float)) * (rededge_band.astype(float) / red_band.astype(float)))  
    tcari[np.isnan(tcari)] = 0  # 将NaN值替换为0  
    mask = np.isinf(tcari)
    max = np.max(tcari[~mask])
    tcari[mask] = max
    return tcari

def calculate_tvi(red_band, green_band, nir_band):  
    """三角植被指数（Triangular Vegetation Index, TVI）"""  
    np.seterr(divide='ignore', invalid='ignore')  
    tvi = 60 * (nir_band.astype(float) - red_band.astype(float)) - 100 * (red_band.astype(float) - green_band.astype(float))  
    tvi[np.isnan(tvi)] = 0  # 将NaN值替换为0  
    mask = np.isinf(tvi)
    max = np.max(tvi[~mask])
    tvi[mask] = max
    return tvi

def calculate_cl1(rededge_band, nir_band):  
    """红边叶绿素指数1（Red-Edge Chlorophyll Index 1, Cl1）"""  
    np.seterr(divide='ignore', invalid='ignore') 
    # Cl1公式: Cl1 = NIR / Rededge - 1
    cl1 = (nir_band.astype(float) / rededge_band.astype(float)) - 1  
    cl1[np.isnan(cl1)] = 0  # 将NaN值替换为0  
    mask = np.isinf(cl1)
    max = np.max(cl1[~mask])
    cl1[mask] = max
    return cl1

def calculate_cl2(rededge_band, green_band):  
    """红边叶绿素指数2（Red-Edge Chlorophyll Index 2, Cl2）"""  
    np.seterr(divide='ignore', invalid='ignore') 
    # Cl2公式: Cl2 = Rededge / Green - 1
    cl2 = (rededge_band.astype(float) / green_band.astype(float)) - 1  
    cl2[np.isnan(cl2)] = 0  # 将NaN值替换为0  
    mask = np.isinf(cl2)
    max = np.max(cl2[~mask])
    cl2[mask] = max
    return cl2    

def calculate_dre(rededge_band, nir_band):  
    """差分红边指数（Difference Red Edge Index, DRE）"""  
    np.seterr(divide='ignore', invalid='ignore')
    # DRE公式: DRE = (NIR - Rededge)
    dre = nir_band.astype(float) - rededge_band.astype(float)
    dre[np.isnan(dre)] = 0  # 将NaN值替换为0  
    mask = np.isinf(dre)
    max = np.max(dre[~mask])
    dre[mask] = max
    return dre

def calculate_ndre(rededge_band, nir_band):  
    """归一化差分红边指数（Normalized Difference Red Edge Index, NDRE）"""  
    np.seterr(divide='ignore', invalid='ignore')
    # NDRE公式: NDRE = (NIR - Rededge) / (NIR + Rededge)
    ndre = (nir_band.astype(float) - rededge_band.astype(float)) / (nir_band.astype(float) + rededge_band.astype(float))
    ndre[np.isnan(ndre)] = 0  # 将NaN值替换为0  
    mask = np.isinf(ndre)
    max = np.max(ndre[~mask])
    ndre[mask] = max
    return ndre

def calculate_sccci(rededge_band, nir_band, red_band):  
    """简化冠层叶绿素含量指数（Simplified Canopy Chlorophyll Content Index, SCCCI）"""  
    np.seterr(divide='ignore', invalid='ignore')
    # SCCCI公式: SCCCI = NDRE / NDVI
    sccci = ((nir_band.astype(float) - rededge_band.astype(float)) / (nir_band.astype(float) + rededge_band.astype(float))) / ((nir_band.astype(float) - red_band.astype(float)) / (nir_band.astype(float) + red_band.astype(float)))
    sccci[np.isnan(sccci)] = 0  # 将NaN值替换为0  
    mask = np.isinf(sccci)
    max = np.max(sccci[~mask])
    sccci[mask] = max
    return sccci

def calculate_msr(nir_band, red_band):  
    """修正简单比值指数（Modified Simple Radio, MSR）"""  
    np.seterr(divide='ignore', invalid='ignore')
    # MSR公式: MSR = (NIR / Red - 1) / ((NIR / Red) ^ 0.5 + 1)
    msr = (nir_band.astype(float) / red_band.astype(float) - 1) / (np.sqrt(nir_band.astype(float) / red_band.astype(float)) + 1)
    msr[np.isnan(msr)] = 0  # 将NaN值替换为0  
    mask = np.isinf(msr)
    max = np.max(msr[~mask])
    msr[mask] = max
    return msr

def calculate_rdvi(nir_band, red_band):  
    """重归一化差异植被指数（renormalized difference vegetation index, RDVI）"""  
    np.seterr(divide='ignore', invalid='ignore')
    # RDVI公式: RDVI = (NIR - Red) / ((NIR + Red) ^ 0.5)
    rdvi = (nir_band.astype(float) - red_band.astype(float)) / np.sqrt(nir_band.astype(float) + red_band.astype(float))
    rdvi[np.isnan(rdvi)] = 0  # 将NaN值替换为0  
    mask = np.isinf(rdvi)
    max = np.max(rdvi[~mask])
    rdvi[mask] = max
    return rdvi

def calculate_gvi(green_band, rededge_band):  
    """绿植被指数（Green vegetation index, GVI）"""  
    np.seterr(divide='ignore', invalid='ignore')
    # GVI公式: GVI = (Green - Rededge) / (Green + Rededge)
    gvi = (green_band.astype(float) - rededge_band.astype(float)) / (green_band.astype(float) + rededge_band.astype(float))
    gvi[np.isnan(gvi)] = 0  # 将NaN值替换为0  
    mask = np.isinf(gvi)
    max = np.max(gvi[~mask])
    gvi[mask] = max
    return gvi

def calculate_mtci(nir_band, red_band, rededge_band):  
    """MERIS陆地叶绿素指数（MERIS Terrestrial Chlorophyll Index, MTCI）"""  
    np.seterr(divide='ignore', invalid='ignore')
    # MTCI公式: MTCI = (NIR - Rededge) / (Rededge - Red)
    mtci = (nir_band.astype(float) - rededge_band.astype(float)) / (rededge_band.astype(float) - red_band.astype(float))
    mtci[np.isnan(mtci)] = 0  # 将NaN值替换为0  
    mask = np.isinf(mtci)
    max = np.max(mtci[~mask])
    mtci[mask] = max
    return mtci

def calculate_nli(nir_band, red_band):  
    """非线性指数（Non Linear Index, NLI）"""  
    np.seterr(divide='ignore', invalid='ignore')
    # NLI公式: NLI = (NIR ^ 2 - Red) / (NIR ^ 2 + Red)
    nli = (np.square(nir_band.astype(float)) - red_band.astype(float)) / (np.square(nir_band.astype(float)) + red_band.astype(float))
    nli[np.isnan(nli)] = 0  # 将NaN值替换为0  
    mask = np.isinf(nli)
    max = np.max(nli[~mask])
    nli[mask] = max
    return nli

def calculate_mnli(nir_band, red_band):  
    """修正非线性指数（Modified Non Linear Index, MNLI）"""  
    np.seterr(divide='ignore', invalid='ignore')
    # MNLI公式: MNLI = (1.5 * (NIR ^ 2 - Red)) / (NIR ^ 2 + Red + 0.5)
    mnli = (1.5 * (np.square(nir_band.astype(float)) - red_band.astype(float))) / (np.square(nir_band.astype(float)) + red_band.astype(float) + 0.5)
    mnli[np.isnan(mnli)] = 0  # 将NaN值替换为0  
    mask = np.isinf(mnli)
    max = np.max(mnli[~mask])
    mnli[mask] = max
    return mnli

def calculate_kNDVI(nir_band, red_band):  
    """核归一化差分植被指数（kernel NDVI, kNDVI）"""  
    np.seterr(divide='ignore', invalid='ignore')
    # NLI公式: kNDVI = tanh(((NIR - Red) / (NIR + Red)) ^ 2)
    kNDVI = np.tanh(np.square((nir_band.astype(float) - red_band.astype(float)) / (nir_band.astype(float) + red_band.astype(float))))
    kNDVI[np.isnan(kNDVI)] = 0  # 将NaN值替换为0  
    mask = np.isinf(kNDVI)
    max = np.max(kNDVI[~mask])
    kNDVI[mask] = max
    return kNDVI

def process_folders(input_folder, output_folder):  
    """遍历输入文件夹下的所有子文件夹，并处理图像以计算植被指数"""  
    for subdir, dirs, files in os.walk(input_folder):  
        if files:  
            # 创建与输入子文件夹同名的输出子文件夹  
            output_subfolder_name = os.path.basename(subdir)  
            output_subfolder = os.path.join(output_folder, output_subfolder_name)  
            if not os.path.exists(output_subfolder):  
                os.makedirs(output_subfolder)  
                  
            # 查找红光、绿光、近红外和红边波段的文件路径  
            red_band_path = find_band_file(subdir, 'result_Red-5')  
            green_band_path = find_band_file(subdir, 'result_Green-5')  
            nir_band_path = find_band_file(subdir, 'result_NIR-5')  
            rededge_band_path = find_band_file(subdir, 'result_RedEdge-5')  
                  
            # 检查是否找到了所有需要的波段文件  
            if red_band_path and green_band_path and nir_band_path and rededge_band_path:  
                # 读取波段数据  
                red_dataset = gdal.Open(red_band_path)  
                green_dataset = gdal.Open(green_band_path)  
                nir_dataset = gdal.Open(nir_band_path)  
                rededge_dataset = gdal.Open(rededge_band_path)  
                      
                if red_dataset is None or green_dataset is None or nir_dataset is None or rededge_dataset is None:  
                    print(f"Failed to open one of the datasets in {subdir}")  
                    continue  
                      
                red_band = red_dataset.GetRasterBand(1).ReadAsArray().astype(float)  
                green_band = green_dataset.GetRasterBand(1).ReadAsArray().astype(float)  
                nir_band = nir_dataset.GetRasterBand(1).ReadAsArray().astype(float)  
                rededge_band = rededge_dataset.GetRasterBand(1).ReadAsArray().astype(float)  
                      
                # 计算植被指数  
                ndvi = calculate_ndvi(red_band, nir_band)
                gndvi = calculate_gndvi(green_band, nir_band)
                rendvi = calculate_rendvi(red_band, rededge_band)
                cig = calculate_cig(green_band, nir_band)
                rgr = calculate_rgr(green_band, red_band)
                rri = calculate_rri(rededge_band, red_band)
                dvi = calculate_dvi(red_band, nir_band)
                savi = calculate_savi(red_band, nir_band)
                osavi = calculate_osavi(red_band, nir_band)  
                mcari = calculate_mcari(red_band, green_band, nir_band)
                tcari = calculate_tcari(red_band, green_band, rededge_band)
                tvi = calculate_tvi(red_band, green_band, nir_band)
                cl1 = calculate_cl1(rededge_band, nir_band)
                cl2 = calculate_cl2(rededge_band, green_band)
                dre = calculate_dre(rededge_band, nir_band)
                ndre = calculate_ndre(rededge_band, nir_band)
                sccci = calculate_sccci(rededge_band, nir_band, red_band)
                msr = calculate_msr(red_band, nir_band)
                rdvi = calculate_rdvi(red_band, nir_band)
                gvi = calculate_gvi(green_band, rededge_band)
                mtci = calculate_mtci(red_band, rededge_band, nir_band)
                nli = calculate_nli(red_band, nir_band)
                mnli = calculate_mnli(red_band, nir_band)
                nrr = calculate_nrr(red_band, nir_band)
                kNDVI = calculate_kNDVI(red_band, nir_band)
               
                # 保存结果  
                output_file_ndvi = os.path.join(output_subfolder, "NDVI-5.tif")
                output_file_gndvi = os.path.join(output_subfolder, "GNDVI-5.tif")
                output_file_rendvi = os.path.join(output_subfolder, "RENDVI-5.tif")
                output_file_cig = os.path.join(output_subfolder, "CIG-5.tif")
                output_file_rgr = os.path.join(output_subfolder, "RGR-5.tif")
                output_file_rri = os.path.join(output_subfolder, "RRI-5.tif")
                output_file_dvi = os.path.join(output_subfolder, "DVI-5.tif")
                output_file_savi = os.path.join(output_subfolder, "SAVI-5.tif")
                output_file_osavi = os.path.join(output_subfolder, "OSAVI-5.tif")
                output_file_mcari = os.path.join(output_subfolder, "MCARI-5.tif")
                output_file_tcari = os.path.join(output_subfolder, "TCARI-5.tif")
                output_file_tvi = os.path.join(output_subfolder, "TVI-5.tif")
                output_file_cl1 = os.path.join(output_subfolder, "Cl1-5.tif")
                output_file_cl2 = os.path.join(output_subfolder, "Cl2-5.tif")
                output_file_dre = os.path.join(output_subfolder, "DRE-5.tif")
                output_file_ndre = os.path.join(output_subfolder, "NDRE-5.tif")
                output_file_sccci = os.path.join(output_subfolder, "SCCCI-5.tif")
                output_file_msr = os.path.join(output_subfolder, "MSR-5.tif")
                output_file_rdvi = os.path.join(output_subfolder, "RDVI-5.tif")
                output_file_gvi = os.path.join(output_subfolder, "GVI-5.tif")
                output_file_mtci = os.path.join(output_subfolder, "MTCI-5.tif")
                output_file_nli = os.path.join(output_subfolder, "NLI-5.tif")
                output_file_mnli = os.path.join(output_subfolder, "MNLI-5.tif")
                output_file_nrr = os.path.join(output_subfolder, "NRR-5.tif")
                output_file_kNDVI = os.path.join(output_subfolder, "kNDVI-5.tif")
                
                driver = gdal.GetDriverByName('GTiff')  
                  
                # 保存NDVI  
                out_dataset_ndvi = driver.Create(output_file_ndvi, nir_dataset.RasterXSize, nir_dataset.RasterYSize, 1, gdal.GDT_Float32)  
                out_dataset_ndvi.GetRasterBand(1).WriteArray(ndvi)  
                out_dataset_ndvi.SetGeoTransform(nir_dataset.GetGeoTransform())  # 复制地理变换信息  
                out_dataset_ndvi.SetProjection(nir_dataset.GetProjection())  # 复制投影信息  
                out_dataset_ndvi.FlushCache()  # 确保数据写入文件  
                out_dataset_ndvi = None  # 关闭数据集
                
                # 保存GNDVI  
                out_dataset_gndvi = driver.Create(output_file_gndvi, nir_dataset.RasterXSize, nir_dataset.RasterYSize, 1, gdal.GDT_Float32)  
                out_dataset_gndvi.GetRasterBand(1).WriteArray(gndvi)  
                out_dataset_gndvi.SetGeoTransform(nir_dataset.GetGeoTransform())  
                out_dataset_gndvi.SetProjection(nir_dataset.GetProjection())  
                out_dataset_gndvi.FlushCache()  
                out_dataset_gndvi = None  
                
                # 保存RENDVI  
                out_dataset_rendvi = driver.Create(output_file_rendvi, nir_dataset.RasterXSize, nir_dataset.RasterYSize, 1, gdal.GDT_Float32)  
                out_dataset_rendvi.GetRasterBand(1).WriteArray(rendvi)  
                out_dataset_rendvi.SetGeoTransform(nir_dataset.GetGeoTransform())  
                out_dataset_rendvi.SetProjection(nir_dataset.GetProjection())  
                out_dataset_rendvi.FlushCache()  
                out_dataset_rendvi = None
                
                # 保存CIG  
                out_dataset_cig = driver.Create(output_file_cig, nir_dataset.RasterXSize, nir_dataset.RasterYSize, 1, gdal.GDT_Float32)  
                out_dataset_cig.GetRasterBand(1).WriteArray(cig)  
                out_dataset_cig.SetGeoTransform(nir_dataset.GetGeoTransform())  
                out_dataset_cig.SetProjection(nir_dataset.GetProjection())  
                out_dataset_cig.FlushCache()  
                out_dataset_cig = None
                
                # 保存RGR  
                out_dataset_rgr = driver.Create(output_file_rgr, nir_dataset.RasterXSize, nir_dataset.RasterYSize, 1, gdal.GDT_Float32)  
                out_dataset_rgr.GetRasterBand(1).WriteArray(rgr)  
                out_dataset_rgr.SetGeoTransform(nir_dataset.GetGeoTransform())  
                out_dataset_rgr.SetProjection(nir_dataset.GetProjection())  
                out_dataset_rgr.FlushCache()  
                out_dataset_rgr = None
                
                # 保存RRI 
                out_dataset_rri = driver.Create(output_file_rri, nir_dataset.RasterXSize, nir_dataset.RasterYSize, 1, gdal.GDT_Float32)  
                out_dataset_rri.GetRasterBand(1).WriteArray(rri)  
                out_dataset_rri.SetGeoTransform(nir_dataset.GetGeoTransform())  
                out_dataset_rri.SetProjection(nir_dataset.GetProjection())  
                out_dataset_rri.FlushCache()  
                out_dataset_rri = None
                
                # 保存DVI  
                out_dataset_dvi = driver.Create(output_file_dvi, nir_dataset.RasterXSize, nir_dataset.RasterYSize, 1, gdal.GDT_Float32)  
                out_dataset_dvi.GetRasterBand(1).WriteArray(dvi)  
                out_dataset_dvi.SetGeoTransform(nir_dataset.GetGeoTransform())  
                out_dataset_dvi.SetProjection(nir_dataset.GetProjection())  
                out_dataset_dvi.FlushCache()  
                out_dataset_dvi = None
                
                # 保存SAVI  
                out_dataset_savi = driver.Create(output_file_savi, nir_dataset.RasterXSize, nir_dataset.RasterYSize, 1, gdal.GDT_Float32)  
                out_dataset_savi.GetRasterBand(1).WriteArray(savi)  
                out_dataset_savi.SetGeoTransform(nir_dataset.GetGeoTransform())  
                out_dataset_savi.SetProjection(nir_dataset.GetProjection())  
                out_dataset_savi.FlushCache()  
                out_dataset_savi = None
                
                # 保存OSAVI  
                out_dataset_osavi = driver.Create(output_file_osavi, nir_dataset.RasterXSize, nir_dataset.RasterYSize, 1, gdal.GDT_Float32)  
                out_dataset_osavi.GetRasterBand(1).WriteArray(osavi)  
                out_dataset_osavi.SetGeoTransform(nir_dataset.GetGeoTransform())  
                out_dataset_osavi.SetProjection(nir_dataset.GetProjection())  
                out_dataset_osavi.FlushCache()  
                out_dataset_osavi = None
                
                # 保存MCARI  
                out_dataset_mcari = driver.Create(output_file_mcari, nir_dataset.RasterXSize, nir_dataset.RasterYSize, 1, gdal.GDT_Float32)  
                out_dataset_mcari.GetRasterBand(1).WriteArray(mcari)  
                out_dataset_mcari.SetGeoTransform(nir_dataset.GetGeoTransform())  
                out_dataset_mcari.SetProjection(nir_dataset.GetProjection())  
                out_dataset_mcari.FlushCache()  
                out_dataset_mcari = None
                
                # 保存TCARI  
                out_dataset_tcari = driver.Create(output_file_tcari, nir_dataset.RasterXSize, nir_dataset.RasterYSize, 1, gdal.GDT_Float32)  
                out_dataset_tcari.GetRasterBand(1).WriteArray(tcari)  
                out_dataset_tcari.SetGeoTransform(nir_dataset.GetGeoTransform())  
                out_dataset_tcari.SetProjection(nir_dataset.GetProjection())  
                out_dataset_tcari.FlushCache()  
                out_dataset_tcari = None
                
                # 保存TVI  
                out_dataset_tvi = driver.Create(output_file_tvi, nir_dataset.RasterXSize, nir_dataset.RasterYSize, 1, gdal.GDT_Float32)  
                out_dataset_tvi.GetRasterBand(1).WriteArray(tvi)  
                out_dataset_tvi.SetGeoTransform(nir_dataset.GetGeoTransform())  
                out_dataset_tvi.SetProjection(nir_dataset.GetProjection())  
                out_dataset_tvi.FlushCache()  
                out_dataset_tvi = None
                
                # 保存Cl1  
                out_dataset_cl1 = driver.Create(output_file_cl1, nir_dataset.RasterXSize, nir_dataset.RasterYSize, 1, gdal.GDT_Float32)  
                out_dataset_cl1.GetRasterBand(1).WriteArray(cl1)  
                out_dataset_cl1.SetGeoTransform(nir_dataset.GetGeoTransform())  
                out_dataset_cl1.SetProjection(nir_dataset.GetProjection())  
                out_dataset_cl1.FlushCache()  
                out_dataset_cl1 = None
                
                # 保存Cl2  
                out_dataset_cl2 = driver.Create(output_file_cl2, nir_dataset.RasterXSize, nir_dataset.RasterYSize, 1, gdal.GDT_Float32)  
                out_dataset_cl2.GetRasterBand(1).WriteArray(cl2)  
                out_dataset_cl2.SetGeoTransform(nir_dataset.GetGeoTransform())  
                out_dataset_cl2.SetProjection(nir_dataset.GetProjection())  
                out_dataset_cl2.FlushCache()  
                out_dataset_cl2 = None
                
                # 保存DRE  
                out_dataset_dre = driver.Create(output_file_dre, nir_dataset.RasterXSize, nir_dataset.RasterYSize, 1, gdal.GDT_Float32)  
                out_dataset_dre.GetRasterBand(1).WriteArray(dre)  
                out_dataset_dre.SetGeoTransform(nir_dataset.GetGeoTransform())  
                out_dataset_dre.SetProjection(nir_dataset.GetProjection())  
                out_dataset_dre.FlushCache()  
                out_dataset_dre = None
                
                # 保存NDRE  
                out_dataset_ndre = driver.Create(output_file_ndre, nir_dataset.RasterXSize, nir_dataset.RasterYSize, 1, gdal.GDT_Float32)  
                out_dataset_ndre.GetRasterBand(1).WriteArray(ndre)  
                out_dataset_ndre.SetGeoTransform(nir_dataset.GetGeoTransform())  
                out_dataset_ndre.SetProjection(nir_dataset.GetProjection())  
                out_dataset_ndre.FlushCache()  
                out_dataset_ndre = None
                
                # 保存SCCCI  
                out_dataset_sccci = driver.Create(output_file_sccci, nir_dataset.RasterXSize, nir_dataset.RasterYSize, 1, gdal.GDT_Float32)  
                out_dataset_sccci.GetRasterBand(1).WriteArray(sccci)  
                out_dataset_sccci.SetGeoTransform(nir_dataset.GetGeoTransform())  # 复制地理变换信息  
                out_dataset_sccci.SetProjection(nir_dataset.GetProjection())  # 复制投影信息  
                out_dataset_sccci.FlushCache()  # 确保数据写入文件  
                out_dataset_sccci = None  # 关闭数据集
                
                # 保存MSR  
                out_dataset_msr = driver.Create(output_file_msr, nir_dataset.RasterXSize, nir_dataset.RasterYSize, 1, gdal.GDT_Float32)  
                out_dataset_msr.GetRasterBand(1).WriteArray(msr)  
                out_dataset_msr.SetGeoTransform(nir_dataset.GetGeoTransform())  # 复制地理变换信息  
                out_dataset_msr.SetProjection(nir_dataset.GetProjection())  # 复制投影信息  
                out_dataset_msr.FlushCache()  # 确保数据写入文件  
                out_dataset_msr = None  # 关闭数据集
                
                # 保存RDVI  
                out_dataset_rdvi = driver.Create(output_file_rdvi, nir_dataset.RasterXSize, nir_dataset.RasterYSize, 1, gdal.GDT_Float32)  
                out_dataset_rdvi.GetRasterBand(1).WriteArray(rdvi)  
                out_dataset_rdvi.SetGeoTransform(nir_dataset.GetGeoTransform())  # 复制地理变换信息  
                out_dataset_rdvi.SetProjection(nir_dataset.GetProjection())  # 复制投影信息  
                out_dataset_rdvi.FlushCache()  # 确保数据写入文件  
                out_dataset_rdvi = None  # 关闭数据集
                
                # 保存GVI  
                out_dataset_gvi = driver.Create(output_file_gvi, nir_dataset.RasterXSize, nir_dataset.RasterYSize, 1, gdal.GDT_Float32)  
                out_dataset_gvi.GetRasterBand(1).WriteArray(gvi)  
                out_dataset_gvi.SetGeoTransform(nir_dataset.GetGeoTransform())  
                out_dataset_gvi.SetProjection(nir_dataset.GetProjection())  
                out_dataset_gvi.FlushCache()  
                out_dataset_gvi = None
                
                # 保存MTCI  
                out_dataset_mtci = driver.Create(output_file_mtci, nir_dataset.RasterXSize, nir_dataset.RasterYSize, 1, gdal.GDT_Float32)  
                out_dataset_mtci.GetRasterBand(1).WriteArray(mtci)  
                out_dataset_mtci.SetGeoTransform(nir_dataset.GetGeoTransform())  
                out_dataset_mtci.SetProjection(nir_dataset.GetProjection())  
                out_dataset_mtci.FlushCache()  
                out_dataset_mtci = None
                
                # 保存NLI  
                out_dataset_nli = driver.Create(output_file_nli, nir_dataset.RasterXSize, nir_dataset.RasterYSize, 1, gdal.GDT_Float32)  
                out_dataset_nli.GetRasterBand(1).WriteArray(nli)  
                out_dataset_nli.SetGeoTransform(nir_dataset.GetGeoTransform())  
                out_dataset_nli.SetProjection(nir_dataset.GetProjection())  
                out_dataset_nli.FlushCache()  
                out_dataset_nli = None
                
                # 保存MNLI  
                out_dataset_mnli = driver.Create(output_file_mnli, nir_dataset.RasterXSize, nir_dataset.RasterYSize, 1, gdal.GDT_Float32)  
                out_dataset_mnli.GetRasterBand(1).WriteArray(mnli)  
                out_dataset_mnli.SetGeoTransform(nir_dataset.GetGeoTransform())  
                out_dataset_mnli.SetProjection(nir_dataset.GetProjection())  
                out_dataset_mnli.FlushCache()  
                out_dataset_mnli = None
                
                # 保存NRR  
                out_dataset_nrr = driver.Create(output_file_nrr, nir_dataset.RasterXSize, nir_dataset.RasterYSize, 1, gdal.GDT_Float32)  
                out_dataset_nrr.GetRasterBand(1).WriteArray(nrr)  
                out_dataset_nrr.SetGeoTransform(nir_dataset.GetGeoTransform())  
                out_dataset_nrr.SetProjection(nir_dataset.GetProjection())  
                out_dataset_nrr.FlushCache()  
                out_dataset_nrr = None
                
                # 保存kNDVI  
                out_dataset_kNDVI = driver.Create(output_file_kNDVI, nir_dataset.RasterXSize, nir_dataset.RasterYSize, 1, gdal.GDT_Float32)  
                out_dataset_kNDVI.GetRasterBand(1).WriteArray(kNDVI)  
                out_dataset_kNDVI.SetGeoTransform(nir_dataset.GetGeoTransform())  
                out_dataset_kNDVI.SetProjection(nir_dataset.GetProjection())  
                out_dataset_kNDVI.FlushCache()  
                out_dataset_kNDVI = None
                                      
                print(f"Processed {subdir} and saved NDVI, GNDVI, RENDVI, CIG, RGR, RRI, DVI, SAVI, OSAVI, MCARI, TCARI, TVI, Cl1, Cl2, DRE, NDRE, SCCCI, MSR, RDVI, GVI, MTCI, NLI, MNLI, NRR, NGR, kNDVI")  
            else:  
                print(f"Missing one or more bands in {subdir}")  
  
input_folder = 'I:\\'  
output_folder = 'F:\\'  
process_folders(input_folder, output_folder)