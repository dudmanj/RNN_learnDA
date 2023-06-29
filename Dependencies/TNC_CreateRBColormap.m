function [mapName] = TNC_CreateRBColormap(numValues,type)
% FUNCTION DETAILS: Simple utility to create a RWB style color map
% _________________________________________________________________________
% PART OF THE TONIC PACKAGE
%   developed by JOSHUA DUDMAN
%   begun at COLUMBIA UNIVERSITY, continuing at HHMI / JFRC
% 
% BUG REPORTING: dudman.web@gmail.com
% CONTRIBUTIONS: people.janelia.org/dudmanj/html/projects.html
% _________________________________________________________________________

switch lower(type)
    
    case 'rb'
        mapLength = numValues;
        halfMap = floor(numValues./2);
        incr = 1./halfMap;
        increaser = (0:incr:1-incr)';
        tmp  = [ increaser, increaser, ones(halfMap,1)];
        tmp2 = [ ones(halfMap,1) , 1-increaser , 1-increaser ];
        mapName = [tmp;tmp2];

    case 'cb'
        mapLength = numValues;
        halfMap = floor(numValues./2);
        incr = 1./halfMap;
        incr2 = 0.33./halfMap;
        increaser = (0:incr:1-incr)';
        increaser2 = (0:incr2:0.33-incr2)';        
        tmp  = [ increaser, 0.67+increaser2, ones(halfMap,1)];
        tmp2 = [ ones(halfMap,1) , 1-increaser , 1-increaser ];
        mapName = [tmp;tmp2];

    case 'wblack'
        incr = 1./numValues;
        increaser = (0:incr:1-incr)';
        tmp  = [ increaser, increaser, increaser ];
        mapName = tmp;
        
    case 'wred'
        incr = 1./numValues;
        increaser = (0:incr:1-incr)';
        tmp  = [ 0.5+(increaser./2) , increaser, increaser ];
        mapName = tmp;

    case 'wblue'
        incr = 1./numValues;
%         increaser = (1-incr:-incr:0)';
        increaser = (0:incr:1-incr)';
%         tmp  = [ increaser, 0.33+(increaser./1.5), 0.5+(increaser./2) ]; % ones(numValues,1) ];
        tmp  = [ increaser , 0.67+0.33*increaser , ones(size(increaser)) ];

        mapName = tmp;

    case 'bo'
        
        if numel(numValues)==2
            incrL = 0.5./abs(numValues(1));
            increaserL = (0:incrL:0.5)';
            tmpL  = [ increaserL , ones(numel(increaserL),1).*0.01, 1-increaserL ];

            incrR = 0.5./abs(numValues(2));
            increaserR = (0.5:incrR:1)';
            tmpR  = [ increaserR , ones(numel(increaserR),1).*0.01, 1-increaserR ];

            mapName = [tmpL ; tmpR];
            
        else
            
            incr = 1./numValues;
            increaser = (0:incr:1-incr)';
            tmp  = [ increaser , ones(numValues,1).*0.5, 1-increaser ];
            mapName = tmp;
            
        end
        
    case 'cpb'
        
        if numel(numValues)==2
            incrL = 0.5./abs(numValues(1));
            increaserL = (0:incrL:0.5)';
            tmpL  = [ increaserL , ones(numel(increaserL),1).*0.67, 1-increaserL ];

            incrR = 0.5./abs(numValues(2));
            increaserR = (0.5:incrR:1)';
            tmpR  = [ increaserR , ones(numel(increaserR),1).*0.67, 1-increaserR ];

            mapName = [tmpL ; tmpR];
            
        else
            
            incr = 1./numValues;
            increaser = (0:incr:1-incr)';
            tmp  = [ increaser , (1-increaser).*0.67, 1-increaser ];
            mapName = tmp;
            
        end        

    case 'cpob'
            
            incr = 1./(numValues/2);
            increaser = (0:incr:1-incr)';
            mid_inc = ([-1:(incr*2):1].^2)';
            tmp  = [ increaser , 1-(0.5*mid_inc(1:numel(increaser))) , 1-increaser ];
            tmp2  = [ 1-increaser , 0.5*mid_inc(1:numel(increaser)) , increaser ];
            mapName = [tmp ; tmp2];
        
    case 'mbr'
        mapName = [ 103,  0, 31;
                    178, 24, 43;
                    214, 96, 77;
                    244,165,130;
                    253,219,199;
                    247,247,247;
                    209,229,240;
                    146,197,222;
                     67,147,195;
                     33,102,172;
                      5, 48, 97 ] ./ 256;
            mapName = flipud(mapName);
                  
    case 'gp'
        mapName = [ 64,  0, 75;
                   118, 42,131;
                   153,112,171;
                   194,165,207;
                   231,212,232;
                   247,247,247;
                   217,240,211;
                   166,219,160;
                    90,174, 97;
                    27,120, 55;
                     0, 68, 27] ./ 256;
                 
                 
    case 'map'
        mapName = [ 166,206,227;
                     31,120,180;
                    178,223,138;
                     51,160, 44;
                    251,154,153;
                    227, 26, 28;
                    253,191,111;
                    255,127,  0;
                    202,178,214;
                    106, 61,154;
                    255,255,153;
                    177, 89, 40] ./ 256;
                
                
    case 'mapb'

        mapName = [ 31,120,180;

                     51,160, 44;
                     
                     106, 61,154;
                     
                     177, 89, 40;

                    227, 26, 28;

                    255,127,  0] ./ 256;  
                
    case 'exag'
%         map1 = TNC_CreateRBColormap(1000,'wblue');
        numValues=1000;
        incr = 1./numValues;
        increaser = (0:incr:1-incr)';
        tmp  = [ increaser , 0.67+0.33*increaser , ones(size(increaser)) ];

        map1 = tmp;
%         map2 = TNC_CreateRBColormap(2000,'cpb');
        numValues=2000;
        incr = 1./numValues;
        increaser = (0:incr:1-incr)';
        tmp  = [ increaser , (1-increaser).*0.67, 1-increaser ];
        map2 = tmp;

        mapName = [map1(1000:-1:1,:) ; map2];

% Sequential yellow blue
    case 'yb'
        mapName = [ 255,255,217;
                    237,248,177;
                    199,233,180;
                    127,205,187;
                    65,182,196;
                    29,145,192;
                    34,94,168;
                    37,52,148;
                    8,29,88 ] ./ 256;

    case 'cat2'

         mapName = [ 141,211,199;
              255,255,179;
              190,186,218;
              251,128,114;
              128,177,211;
              253,180,98;
              179,222,105;
              252,205,229;
              217,217,217;
              188,128,189;
              204,235,197;
              255,237,111] ./ 256;             
        
end
