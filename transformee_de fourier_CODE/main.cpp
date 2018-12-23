
 		

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
   //											                //				
  //      PROGRAMME POUR LES TRAITEMENTS FREQUENTIELS ET DE LA TRANSFORMEE DE FOURIER DES IMAGES       //
 //			Auteur OUBDA Raphael Nicolas Wendyam					      //
///////////////////////////////////////////////////////////////////////////////////////////////////////
//Réferrence: http://docs.opencv.org/doc/tutorials/core/discrete_fourier_transform/discrete_fourier_transform.html.*/

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat TansFourier (Mat imgorig);
Mat SpecTansFourier (Mat imageTransFourier);
Mat TansFourierInverse (Mat imageTransFourier, int nblignes, int nbcolones);

    //////////////////////////////////////////////////////////////////////////////////////////////
   //											       //				
  //      	FONCTIONS UTILISEES POUR LE TRAITEMENT FREQUENTIEL			      //
 //											     //
//////////////////////////////////////////////////////////////////////////////////////////////

Mat OptimalSize (Mat I){

	//image redimensionnee  a renvoyer en sortie	    
	Mat padded;     

	// taille optimale de l image redimensionnee
	int m = getOptimalDFTSize( I.rows );
	int n = getOptimalDFTSize( I.cols );

	//Ajout de valeur de bourrage dans l'image redimensionnee
	copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));
	return padded;
}

Mat Fourier (Mat RedimSize){

	// Creation de deux Conteneurs avec valeurs flottantes pour les parties reelles et imaginaires de la transformee de fourier
	Mat planes[] = {Mat_<float>(RedimSize), Mat::zeros(RedimSize.size(), CV_32F)};

	//Concatenation des deux conteneurs
	Mat complexI;
	merge(planes, 2, complexI);

	//Transformée de fourier
	dft(complexI, complexI);
	
	return complexI;
}

Mat Norm (Mat complexI){

	vector<Mat> planes;
	// separation des parties reelles et imaginaires
	split(complexI, planes);

	// Calcul de norme de la transformee de fourier
	magnitude(planes[0], planes[1], planes[0]);
	Mat MagI = planes[0];

	// Passage a l'echelle logarithmique
	MagI += Scalar::all(1);
	log(MagI, MagI);

	// recadrage du spectre
        MagI = MagI(Rect(0, 0, MagI.cols & -2, MagI.rows & -2));
	normalize(MagI,MagI, 0, 1, CV_MINMAX);
	Mat Out;
	MagI.convertTo(Out, CV_8UC1,255);
	
	return Out;
}

Mat Newcadran(Mat MagI){
	
	int cx = MagI.cols/2;
        int cy = MagI.rows/2;
	
	// Top-Left - Create a ROI per quadrant
        Mat q0(MagI, Rect(0, 0, cx, cy));  

	 // Top-Right 
        Mat q1(MagI, Rect(cx, 0, cx, cy));
	
	// Bottom-Left 
	Mat q2(MagI, Rect(0, cy, cx, cy));  

	// Bottom-Right
        Mat q3(MagI, Rect(cx, cy, cx, cy)); 
	
	// swap quadrants (Top-Left with Bottom-Right)
        Mat tmp;                          
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);
	
	// swap quadrant (Top-Right with Bottom-Left)
        q1.copyTo(tmp);                    
        q2.copyTo(q1);
        tmp.copyTo(q2);

    //normalize(MagI, MagI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
                                            // viewable image form (float between values 0 and 1).
//Reconstruction de l'image par la dft inverse

	return MagI;
}

Mat  Invertdft (Mat complexI, int n_rows, int n_cols){
	Mat ReconsImag;
	Mat InvertdftImag;
	vector<Mat> planes;

	//Transformee inverse de fourier
	//dft( ImageTransFourier,ImageFourierInverse,DFT_INVERSE + DFT_SCALE);
	dft(complexI, InvertdftImag,DFT_INVERSE+ DFT_SCALE);

	split(InvertdftImag,planes);

	ReconsImag = planes[0];

	//Recadrage de l image
	ReconsImag = ReconsImag(Rect(0, 0, ReconsImag.cols & -2, ReconsImag.rows & -2));

	//Conversion en image à niveaux de gris
	ReconsImag.convertTo(ReconsImag, CV_8UC1);
	Mat Out (ReconsImag, Rect(0,0, n_cols, n_rows));

	return Out;
}

Mat LowpassFilter (Mat complexI, float cut_freq ){
	
	Mat LowpassImag;
	vector<Mat> planes;
	float cx = complexI.cols / (float) 2;
	float cy = complexI.rows / (float) 2;
		
	// Determination du rayon du cercle
	int rayon = (int) (min(float(cx), float(cy)))*cut_freq;
	split(complexI,planes);

	// Mise à jour des valeurs à l'extérieur du cercle
	for(int i = 0;  i< (int) planes.size(); i++){
		Newcadran(planes[i]);
		for(int j = 0; j < planes[i].rows; j++)
		for(int w = 0; w< planes[i].cols; w++)
			{
				if(pow((w - cx),2) + pow((j - cy),2) > pow(rayon,2))
					planes[i].at<float>(j,w) = 0;
			}
	}
	merge(planes,LowpassImag);
	return LowpassImag;
}

Mat HightpassFilter (Mat complexI, float cut_freq ){
	    
	Mat HightpassImag;
	vector<Mat> planes;
	float cx = complexI.cols / (float) 2;
	float cy = complexI.rows / (float) 2;
		
	// Determination du rayon du cercle
	int rayon = (int) (min(float(cx), float(cy)))*cut_freq;
	split(complexI,planes);
		
	// Mise à jour des valeurs à l'extérieur du cercle
	for(int i = 0;  i< (int) planes.size(); i++){
		Newcadran(planes[i]);
		for(int j = 0; j < planes[i].rows; j++)
			for(int w = 0; w< planes[i].cols; w++)
				{
					if(pow((w - cx),2) + pow((j - cy),2)  < pow(rayon,2))
						planes[i].at<float>(j,w) = 0;
				}
			}
	merge(planes,HightpassImag);
	return HightpassImag;
}


    //////////////////////////////////////////////////////////////////////////////////////////// 
   //											     //				
  //      			PROGRAMME PRINCIPALE			  		    //
 //											   //
////////////////////////////////////////////////////////////////////////////////////////////


int main(int argc, char ** argv)
{
	
    //DECLARATION DES VARIABLES
	char nomimg[100];
	Mat imageInitiale;
	Mat imageSpectre;
	Mat imageTransFourier;
	Mat imageFinale;
	int choixF;
	float cut_freq;
	int choix;
	cout << "*****************************************************************"<<endl;
	cout << "*                                                               *"<<endl;
	cout << "*     TRANSFORMEE DE FOURIER ET TRAITEMENTS FREQUENCIELS        *"<<endl;
	cout << "* 	       					                 *"<<endl;
	cout << "*****************************************************************"<<endl;
	cout << "                                                                   "<<endl;
	cout << "                                                                   "<<endl;
	cout<<"*******1---Pour la Tansformée de Fourier**********************"<<endl;
	cout<<"*******2---Pour les traitements fréqentiels **********************"<<endl;
	cout<<" VOTRE CHOIX:  ";
	cin>> choix;

	if(choix==1){
		if(argc!=1){
			cout << "Saisir seulement le nom du programme : ./DFT "<< endl;
		}
		else{
			cout << "Transformée de FOURIER d'image"<<endl;
			cout << "Donner le nom de votre image y compris avec l'extension (.png,.tif,.jpg...) et son chemin d'acces"<<endl;
	       		cout << "$ ";
			cin  >> nomimg;

			//Chargement de l'image originale en convertissant en niveau de gris
			imageInitiale = imread(nomimg, CV_LOAD_IMAGE_GRAYSCALE); 

	    		if(!imageInitiale.data){
				cout << "Merci de fournir une image valide \n"<<endl;
	   		}
	    		else{

				//Transformée de Fourrier
				imageTransFourier = TansFourier (imageInitiale);

				//Spectre de Fourrier
				imageSpectre = SpecTansFourier (imageTransFourier);

				//Inverse de la transformée
				imageFinale = TansFourierInverse (imageTransFourier, imageInitiale.rows, imageInitiale.cols);

				// Affichages des résultats
				imshow("Image initiale en niveau de gris" , imageInitiale);
				imshow("Spectre de FOURIER", imageSpectre);
				imshow("Image restauree", imageFinale);

	       			 // Enregistrement de l'image initiale
				 if(!imwrite("Image_initiale.png", imageInitiale))
				 cout<<"Erreur lors de l'enregistrement"<<endl;

				// Enregistrement de l'image du spectre
				if(!imwrite("Spectre.png", imageSpectre))
				cout<<"Erreur lors de l'enregistrement"<<endl;

				// Enregistrement de l'image de la transformée
				if(!imwrite("imageRestituee.png", imageFinale))
				cout<<"Erreur lors de l'enregistrement"<<endl;
			
				waitKey();
	       			return 0;
	    		}
	  	 }
	}
	else{
		
		cout << "Donner le nom de votre image y compris avec l'extension (.png,.tif,.jpg...) et son chemin d'acces"<<endl;
	       	cout << "$ ";
		cin  >> nomimg;
		
		//Chargement de l'image originale en convertissant en niveau de gris
		Mat I = imread(nomimg, CV_LOAD_IMAGE_GRAYSCALE); 

		cout << "Choix du filtre"<<endl;
		cout << "Tapez 1 pour le filtre passe-bas"<<endl;
		cout << "Tapez 2 pour le filtre passe-haut"<<endl;
		cout << "choix :";
		cin >> choixF;
		cout << "Entrer la frequence de coupure 0 < fc <= 1:";
		cin >> cut_freq;
		const char* filename = argv[1];

		// Transforme l'image d'entrée en gris
		//Mat I = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

		if(!I.data){
			cout << "Veuillez fournir une image valide"<<endl;
			return -1;
		}
		else{

			// Calcul de la transformée de fourier de l'image
			Mat complexI = Fourier(OptimalSize(I));
			Mat FourierImg;

			//Application du filtre suivant le choix
			if(choixF ==2){
				FourierImg = HightpassFilter(complexI, cut_freq);
				
			}
			else if(choixF ==1){
				FourierImg = LowpassFilter(complexI, cut_freq);

			}

			//Détermination de la norme du spectre
			Mat Spectrum1 = Newcadran(Norm(complexI));
			Mat EndSpectrum = Norm(FourierImg);
			Mat InvertdftImag = Invertdft(Newcadran(FourierImg),I.rows, I.cols);

			//Affichage des différentes images
			imshow("Image initiale", I);
			imshow("Spectre de Fourier Initial", Spectrum1);
			imshow("Spectre de Fourier Traite", EndSpectrum);
			imshow("Image Apres Traitement", InvertdftImag);

			// Enregistrement 
			if(!imwrite("ImgOrignie.png", I))
				cout<<"Erreur"<<endl;
			if(!imwrite("SpectreFourier.png", Spectrum1))
				cout<<"Erreur"<<endl;
			if(!imwrite("Cut_Frq.png", EndSpectrum))
				cout<<"Erreur"<<endl;
			if(!imwrite("RestoredImg.png",InvertdftImag))
				cout<<"Erreur"<<endl;
			waitKey();
			return 0;
		}
	}
}

    ////////////////////////////////////////////////////////////////////////////////////////////
   //											     //				
  //      	FONCTIONS UTILISEES POUR LA TRANSFORMEE DE FOURIER			    //
 //											   //
////////////////////////////////////////////////////////////////////////////////////////////

Mat TansFourier (Mat imgorig){
	
	Mat imgredim;

        //Redimentionnement de l'image d'entrée à la taille optimale
        int m = getOptimalDFTSize( imgorig.rows );
        int n = getOptimalDFTSize( imgorig.cols );

        // Formation des bordures de l'image source en les mettant a zero
        copyMakeBorder(imgorig, imgredim, 0, m - imgorig.rows, 0, n - imgorig.cols, BORDER_CONSTANT, Scalar::all(0));

        //Stockage dans un format flottant de l'image d'entrée redimentionnemt avec canal pour contenir les valeurs complexes
        Mat planes[] = {Mat_<float>(imgredim), Mat::zeros(imgredim.size(), CV_32F)};
        Mat complexI;
	
	// Completer le plan redimentinné avec des zéros
        merge(planes, 2, complexI);  

        //Transformée de FOURTIER
        dft(complexI, complexI);     

        return complexI;
}

// Fonction du calcul du spectre
Mat SpecTansFourier (Mat imageTransFourier){

        vector<Mat> planes;
        // separation des parties reelles et imaginaires
        split(imageTransFourier, planes);   

        // Calcul de norme de la transformée de Fourier
        magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
        Mat magI = planes[0];

        // Passage à l'échelle logarithmique
        magI += Scalar::all(1);
        log(magI, magI);

        // Recadrage du spectre, s'il a un nombre impair de lignes ou de colonnes
        magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

        // Réorganisation des quadrants de l'image de Fourier afin que l'origine soit au centre de l'image
        int cx = magI.cols/2;
        int cy = magI.rows/2;
	
	// En haut à gauche
        Mat q0(magI, Rect(0, 0, cx, cy)); 

	// En haut à droite
        Mat q1(magI, Rect(cx, 0, cx, cy)); 

	// En bas à gauche 
        Mat q2(magI, Rect(0, cy, cx, cy)); 
 
	// En bas à droite
        Mat q3(magI, Rect(cx, cy, cx, cy)); 
	
	// Quadrants d'échange (Haut-Gauche avec Bas-Droite)
        Mat tmp;                           
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);
	
	// Quadruple d'échange (Haut-Droite avec Bas-Gauche)
        q1.copyTo(tmp);                    
        q2.copyTo(q1);
        tmp.copyTo(q2);

        //Transformez la matrice en valeurs flottantes en une forme d'image visible (float entre les valeurs 0 et 1).
        normalize(magI, magI, 0, 1, CV_MINMAX);

        return magI;
}

// Fonction de la transformée de Fourier inversée
Mat TansFourierInverse (Mat imageSpectre, int rows, int cols){
	Mat imageFinale;
	Mat imgFourierInverse;
	vector<Mat> planes;

	//Transformée inverse de Fourier
	dft( imageSpectre,imgFourierInverse,DFT_INVERSE + DFT_SCALE);

	split(imgFourierInverse,planes);

	imageFinale = planes[0];

	//Recadrage de l'image
	imageFinale = imageFinale(Rect(0, 0, imageFinale.cols & -2, imageFinale.rows & -2));

	//Conversion en image à niveaux de gris
	imageFinale.convertTo(imageFinale, CV_8UC1);
	Mat imgInvTransFour (imageFinale, Rect(0,0, rows, cols));

	return imgInvTransFour;
}

