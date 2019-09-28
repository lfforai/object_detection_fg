#pragma once
#include "image.h"
//#include "error_util.h"

#define IMAGE_H 28
#define IMAGE_W 28

template<class value_type>
image<value_type>::image(){ 
  
}

template<class value_type>
image<value_type>::~image(){
 
}

void FreeImageErrorHandler(FREE_IMAGE_FORMAT oFif, const char *zMessage)
{
	FatalError(zMessage);
}

template<class value_type>
void image<value_type>::readImage(const char* fname)
{
	// declare a host image object for an 8-bit grayscale image
	std::string sFilename(fname);
	std::cout << "Loading image " << sFilename << std::endl;

	//Take care of half precision
	Convert<value_type> fromReal;

	// load gray-scale image from disk    
	// set your own FreeImage error handler
	FreeImage_SetOutputMessage(FreeImageErrorHandler);

	FREE_IMAGE_FORMAT eFormat = FreeImage_GetFileType(sFilename.c_str());//根据图片文件后缀名字，得到图片的类型

	// no signature? try to guess the file format from the file extension
	if (eFormat == FIF_UNKNOWN)
	{
		eFormat = FreeImage_GetFIFFromFilename(sFilename.c_str());
	}

	if (eFormat == FIF_UNKNOWN) //推断文件类型
	{
		FatalError("Unknown image format");
	}
	// check that the plugin has reading capabilities ...

	FIBITMAP *pBitmap;
	if (FreeImage_FIFSupportsReading(eFormat))
	{
		pBitmap = FreeImage_Load(eFormat, sFilename.c_str());
	}

	if (pBitmap == 0)
	{
		FatalError("Error reading image");
	}
	
	if(FreeImage_GetImageType(pBitmap) != FIT_BITMAP)
	 {
	    FatalError("This is not FIT_BITMAP imagee");
     }
	
	//create an ImageCPU to receive the loaded image data
	//ImageCPU_8u_C1 oImage(FreeImage_GetWidth(pBitmap), FreeImage_GetHeight(pBitmap));

	int width = FreeImage_GetWidth(pBitmap);
	int height = FreeImage_GetHeight(pBitmap);
	int channl_o = FreeImage_GetBPP(pBitmap);

    channl_o = (int)(channl_o / 8);
	this->channl = channl_o;
	this->width = width;
	this->height = height;
	this->size= channl_o*height*width;

	this->imgData_h =(value_type* )malloc(sizeof(value_type)*this->size);

	//Normalize image to be in range [0,1]
	for (int i = 0; i < height; ++i)
	{
		unsigned char *pSrcLine = FreeImage_GetScanLine(pBitmap,height - i - 1);
		for (int j = 0; j < width*channl_o; j++)
		{
			int idx = width*channl_o * i + j;
			this->imgData_h[idx] = fromReal(*(pSrcLine + j)/(double)255);
		}
	}

	//Mat ima=cv::imread(sFilename);
	//ima.convertTo(ima, CV_32F);
	//float* p = ima.ptr<float>();
	FreeImage_Unload(pBitmap); //用完就抛弃掉
}

template<class value_type>
void image<value_type>::printDeviceVector(int size, value_type* vec_d)
{
	typedef typename ScaleFactorTypeMap<value_type>::Type real_type; //专门为了half1 设计的，转float
	value_type *vec;
	vec = new value_type[size];//在堆上放置数据
	cudaDeviceSynchronize(); //目的是如果还有device上的运算没有完成，等待其完成运算
	cudaMemcpy(vec, vec_d, size * sizeof(value_type), cudaMemcpyDeviceToHost); //拷贝到host上
	Convert<real_type> toReal;//转换为可以打印的支持类型，float=float double=double ，half1=flaot
	std::cout.precision(7); //精度转换的功能
	std::cout.setf(std::ios::fixed, std::ios::floatfield);
	for (int i = 0; i < size; i++)
	{
		std::cout << toReal(vec[i]) << " ";//打印结果
	}
	std::cout << std::endl;
	delete[] vec;
}

template class image<float>;
template class image<half1>;
