import numpy as np

element_parameters = {
    'H':     [np.array([0.489918, 0.262003, 0.196767, 0.049879]),
              np.array([20.659300, 7.740390, 49.551900, 2.201590]),
              0.001305],
    'H1-':   [np.array([0.897661, 0.565616, 0.415815, 0.116973]),
              np.array([53.136800, 15.187000, 186.576000, 3.567090]),
              0.002389],
    'He':    [np.array([0.873400, 0.630900, 0.311200, 0.178000]),
              np.array([9.103700, 3.356800, 22.927600, 0.982100]),
              0.006400],
    'Li':    [np.array([1.128200, 0.750800, 0.617500, 0.465300]),
              np.array([3.954600, 1.052400, 85.390500, 168.261000]),
              0.037700],
    'Li1+':  [np.array([0.696800, 0.788800, 0.341400, 0.156300]),
              np.array([4.623700, 1.955700, 0.631600, 10.095300]),
              0.016700],
    'Be':    [np.array([1.591900, 1.127800, 0.539100, 0.702900]),
              np.array([43.642700, 1.862300, 103.483000, 0.542000]),
              0.038500],
    'Be2+':  [np.array([6.260300, 0.884900, 0.799300, 0.164700]),
              np.array([0.002700, 0.831300, 2.275800, 5.114600]),
              -6.109200],
    'B':     [np.array([2.054500, 1.332600, 1.097900, 0.706800]),
              np.array([23.218500, 1.021000, 60.349800, 0.140300]),
              -0.193200],
    'C':     [np.array([2.310000, 1.020000, 1.588600, 0.865000]),
              np.array([20.843900, 10.207500, 0.568700, 51.651200]),
              0.215600],
    'Cval':  [np.array([2.260690, 1.561650, 1.050750, 0.839259]),
              np.array([22.690700, 0.656665, 9.756180, 55.594900]),
              0.286977],
    'N':     [np.array([12.212600, 3.132200, 2.012500, 1.166300]),
              np.array([0.005700, 9.893300, 28.997500, 0.582600]),
              -11.529000],
    'O':     [np.array([3.048500, 2.286800, 1.546300, 0.867000]),
              np.array([13.277100, 5.701100, 0.323900, 32.908900]),
              0.250800],
    'O1-':   [np.array([4.191600, 1.639690, 1.526730, -20.307000]),
              np.array([12.857300, 4.172360, 47.017900, -0.014040]),
              21.941200],
    'F':     [np.array([3.539200, 2.641200, 1.517000, 1.024300]),
              np.array([10.282500, 4.294400, 0.261500, 26.147600]),
              0.277600],
    'F1-':   [np.array([3.632200, 3.510570, 1.260640, 0.940706]),
              np.array([5.277560, 14.735300, 0.442258, 47.343700]),
              0.653396],
    'Ne':    [np.array([3.955300, 3.112500, 1.454600, 1.125100]),
              np.array([8.404200, 3.426200, 0.230600, 21.718400]),
              0.351500],
    'Na':    [np.array([4.762600, 3.173600, 1.267400, 1.112800]),
              np.array([3.285000, 8.842200, 0.313600, 129.424000]),
              0.676000],
    'Na1+':  [np.array([3.256500, 3.936200, 1.399800, 1.003200]),
              np.array([2.667100, 6.115300, 0.200100, 14.039000]),
              0.404000],
    'Mg':    [np.array([5.420400, 2.173500, 1.226900, 2.307300]),
              np.array([2.827500, 79.261100, 0.380800, 7.193700]),
              0.858400],
    'Mg2+':  [np.array([3.498800, 3.837800, 1.328400, 0.849700]),
              np.array([2.167600, 4.754200, 0.185000, 10.141100]),
              0.485300],
    'Al':    [np.array([6.420200, 1.900200, 1.593600, 1.964600]),
              np.array([3.038700, 0.742600, 31.547200, 85.088600]),
              1.115100],
    'Al3+':  [np.array([4.174480, 3.387600, 1.202960, 0.528137]),
              np.array([1.938160, 4.145530, 0.228753, 8.285240]),
              0.706786],
    'Siv':   [np.array([6.291500, 3.035300, 1.989100, 1.541000]),
              np.array([2.438600, 32.333700, 0.678500, 81.693700]),
              1.140700],
    'Sival': [np.array([5.662690, 3.071640, 2.624460, 1.393200]),
              np.array([2.665200, 38.663400, 0.916946, 93.545800]),
              1.247070],
    'Si4+':  [np.array([4.439180, 3.203450, 1.194530, 0.416530]),
              np.array([1.641670, 3.437570, 0.214900, 6.653650]),
              0.746297],
    'P':     [np.array([6.434500, 4.179100, 1.780000, 1.490800]),
              np.array([1.906700, 27.157000, 0.526000, 68.164500]),
              1.114900],
    'S':     [np.array([6.905300, 5.203400, 1.437900, 1.586300]),
              np.array([1.467900, 22.215100, 0.253600, 56.172000]),
              0.866900],
    'Cl':    [np.array([11.460400, 7.196400, 6.255600, 1.645500]),
              np.array([0.010400, 1.166200, 18.519400, 47.778400]),
              -9.557400],
    'Cl1-':  [np.array([18.291500, 7.208400, 6.533700, 2.338600]),
              np.array([0.006600, 1.171700, 19.542400, 60.448600]),
              -16.378000],
    'Ar':    [np.array([7.484500, 6.772300, 0.653900, 1.644200]),
              np.array([0.907200, 14.840700, 43.898300, 33.392900]),
              1.444500],
    'K':     [np.array([8.218600, 7.439800, 1.051900, 0.865900]),
              np.array([12.794900, 0.774800, 213.187000, 41.684100]),
              1.422800],
    'K1+':   [np.array([7.957800, 7.491700, 6.359000, 1.191500]),
              np.array([12.633100, 0.767400, -0.002000, 31.912800]),
              -4.997800],
    'Ca':    [np.array([8.626600, 7.387300, 1.589900, 1.021100]),
              np.array([10.442100, 0.659900, 85.748400, 178.437000]),
              1.375100],
    'Ca2+':  [np.array([15.634800, 7.951800, 8.437200, 0.853700]),
              np.array([-0.007400, 0.608900, 10.311600, 25.990500]),
              -14.875000],
    'Sc':    [np.array([9.189000, 7.367900, 1.640900, 1.468000]),
              np.array([9.021300, 0.572900, 136.108000, 51.353100]),
              1.332900],
    'Sc3+':  [np.array([13.400800, 8.027300, 1.659430, 1.579360]),
              np.array([0.298540, 7.962900, -0.286040, 16.066200]),
              -6.666700],
    'Ti':    [np.array([9.759500, 7.355800, 1.699100, 1.902100]),
              np.array([7.850800, 0.500000, 35.633800, 116.105000]),
              1.280700],
    'Ti2+':  [np.array([9.114230, 7.621740, 2.279300, 0.087899]),
              np.array([7.524300, 0.457585, 19.536100, 61.655800]),
              0.897155],
    'Ti3+':  [np.array([17.734400, 8.738160, 5.256910, 1.921340]),
              np.array([0.220610, 7.047160, -0.157620, 15.976800]),
              -14.652000],
    'Ti4+':  [np.array([19.511400, 8.234730, 2.013410, 1.520800]),
              np.array([0.178847, 6.670180, -0.292630, 12.946400]),
              -13.280000],
    'V':     [np.array([10.297100, 7.351100, 2.070300, 2.057100]),
              np.array([6.865700, 0.438500, 26.893800, 102.478000]),
              1.219900],
    'V2+':   [np.array([10.106000, 7.354100, 2.288400, 0.022300]),
              np.array([6.881800, 0.440900, 20.300400, 115.122000]),
              1.229800],
    'V3+':   [np.array([9.431410, 7.741900, 2.153430, 0.016865]),
              np.array([6.395350, 0.383349, 15.190800, 63.969000]),
              0.656565],
    'V5+':   [np.array([15.688700, 8.142080, 2.030810, -9.576000]),
              np.array([0.679003, 5.401350, 9.972780, 0.940464]),
              1.714300],
    'Cr':    [np.array([10.640600, 7.353700, 3.324000, 1.492200]),
              np.array([6.103800, 0.392000, 20.262600, 98.739900]),
              1.183200],
    'Cr2+':  [np.array([9.540340, 7.750900, 3.582740, 0.509107]),
              np.array([5.660780, 0.344261, 13.307500, 32.422400]),
              0.616898],
    'Cr3+':  [np.array([9.680900, 7.811360, 2.876030, 0.113575]),
              np.array([5.594630, 0.334393, 12.828800, 32.876100]),
              0.518275],
    'Mn':    [np.array([11.281900, 7.357300, 3.019300, 2.244100]),
              np.array([5.340900, 0.343200, 17.867400, 83.754300]),
              1.089600],
    'Mn2+':  [np.array([10.806100, 7.362000, 3.526800, 0.218400]),
              np.array([5.279600, 0.343500, 14.343000, 41.323500]),
              1.087400],
    'Mn3+':  [np.array([9.845210, 7.871940, 3.565310, 0.323613]),
              np.array([4.917970, 0.294393, 10.817100, 24.128100]),
              0.393974],
    'Mn4+':  [np.array([9.962530, 7.970570, 2.760670, 0.054447]),
              np.array([4.848500, 0.283303, 10.485200, 27.573000]),
              0.251877],
    'Fe':    [np.array([11.769500, 7.357300, 3.522200, 2.304500]),
              np.array([4.761100, 0.307200, 15.353500, 76.880500]),
              1.036900],
    'Fe2+':  [np.array([11.042400, 7.374000, 4.134600, 0.439900]),
              np.array([4.653800, 0.305300, 12.054600, 31.280900]),
              1.009700],
    'Fe3+':  [np.array([11.176400, 7.386300, 3.394800, 0.072400]),
              np.array([4.614700, 0.300500, 11.672900, 38.556600]),
              0.970700],
    'Co':    [np.array([12.284100, 7.340900, 4.003400, 2.348800]),
              np.array([4.279100, 0.278400, 13.535900, 71.169200]),
              1.011800],
    'Co2+':  [np.array([11.229600, 7.388300, 4.739300, 0.710800]),
              np.array([4.123100, 0.272600, 10.244300, 25.646600]),
              0.932400],
    'Co3+':  [np.array([10.338000, 7.881730, 4.767950, 0.725591]),
              np.array([3.909690, 0.238668, 8.355830, 18.349100]),
              0.286667],
    'Ni':    [np.array([12.837600, 7.292000, 4.443800, 2.380000]),
              np.array([3.878500, 0.256500, 12.176300, 66.342100]),
              1.034100],
    'Ni2+':  [np.array([11.416600, 7.400500, 5.344200, 0.977300]),
              np.array([3.676600, 0.244900, 8.873000, 22.162600]),
              0.861400],
    'Ni3+':  [np.array([10.780600, 7.758680, 5.227460, 0.847114]),
              np.array([3.547700, 0.223140, 7.644680, 16.967300]),
              0.386044],
    'Cu':    [np.array([13.338000, 7.167600, 5.615800, 1.673500]),
              np.array([3.582800, 0.247000, 11.396600, 64.812600]),
              1.191000],
    'Cu1+':  [np.array([11.947500, 7.357300, 6.245500, 1.557800]),
              np.array([3.366900, 0.227400, 8.662500, 25.848700]),
              0.890000],
    'Cu2+':  [np.array([11.816800, 7.111810, 5.781350, 1.145230]),
              np.array([3.374840, 0.244078, 7.987600, 19.897000]),
              1.144310],
    'Zn':    [np.array([14.074300, 7.031800, 5.165200, 2.410000]),
              np.array([3.265500, 0.233300, 10.316300, 58.709700]),
              1.304100],
    'Zn2+':  [np.array([11.971900, 7.386200, 6.466800, 1.394000]),
              np.array([2.994600, 0.203100, 7.082600, 18.099500]),
              0.780700],
    'Ga':    [np.array([15.235400, 6.700600, 4.359100, 2.962300]),
              np.array([3.066900, 0.241200, 10.780500, 61.413500]),
              1.718900],
    'Ga3+':  [np.array([12.692000, 6.698830, 6.066920, 1.006600]),
              np.array([2.812620, 0.227890, 6.364410, 14.412200]),
              1.535450],
    'Ge':    [np.array([16.081600, 6.374700, 3.706800, 3.683000]),
              np.array([2.850900, 0.251600, 11.446800, 54.762500]),
              2.131300],
    'Ge4+':  [np.array([12.917200, 6.700030, 6.067910, 0.859041]),
              np.array([2.537180, 0.205855, 5.479130, 11.603000]),
              1.455720],
    'As':    [np.array([16.672300, 6.070100, 3.431300, 4.277900]),
              np.array([2.634500, 0.264700, 12.947900, 47.797200]),
              2.531000],
    'Se':    [np.array([17.000600, 5.819600, 3.973100, 4.354300]),
              np.array([2.409800, 0.272600, 15.237200, 43.816300]),
              2.840900],
    'Br':    [np.array([17.178900, 5.235800, 5.637700, 3.985100]),
              np.array([2.172300, 16.579600, 0.260900, 41.432800]),
              2.955700],
    'Br1-':  [np.array([17.171800, 6.333800, 5.575400, 3.727200]),
              np.array([2.205900, 19.334500, 0.287100, 58.153500]),
              3.177600],
    'Kr':    [np.array([17.355500, 6.728600, 5.549300, 3.537500]),
              np.array([1.938400, 16.562300, 0.226100, 39.397200]),
              2.825000],
    'Rb':    [np.array([17.178400, 9.643500, 5.139900, 1.529200]),
              np.array([1.788800, 17.315100, 0.274800, 164.934000]),
              3.487300],
    'Rb1+':  [np.array([17.581600, 7.659800, 5.898100, 2.781700]),
              np.array([1.713900, 14.795700, 0.160300, 31.208700]),
              2.078200],
    'Sr':    [np.array([17.566300, 9.818400, 5.422000, 2.669400]),
              np.array([1.556400, 14.098800, 0.166400, 132.376000]),
              2.506400],
    'Sr2+':  [np.array([18.087400, 8.137300, 2.565400, -34.193000]),
              np.array([1.490700, 12.696300, 24.565100, -0.013800]),
              41.402500],
    'Y':     [np.array([17.776000, 10.294600, 5.726290, 3.265880]),
              np.array([1.402900, 12.800600, 0.125599, 104.354000]),
              1.912130],
    'Y3+':   [np.array([17.926800, 9.153100, 1.767950, -33.108000]),
              np.array([1.354170, 11.214500, 22.659900, -0.013190]),
              40.260200],
    'Zr':    [np.array([17.876500, 10.948000, 5.417320, 3.657210]),
              np.array([1.276180, 11.916000, 0.117622, 87.662700]),
              2.069290],
    'Zr4+':  [np.array([18.166800, 10.056200, 1.011180, -2.647900]),
              np.array([1.214800, 10.148300, 21.605400, -0.102760]),
              9.414540],
    'Nb':    [np.array([17.614200, 12.014400, 4.041830, 3.533460]),
              np.array([1.188650, 11.766000, 0.204785, 69.795700]),
              3.755910],
    'Nb3+':  [np.array([19.881200, 18.065300, 11.017700, 1.947150]),
              np.array([0.019175, 1.133050, 10.162100, 28.338900]),
              -12.912000],
    'Nb5+':  [np.array([17.916300, 13.341700, 10.799000, 0.337905]),
              np.array([1.124460, 0.028781, 9.282060, 25.722800]),
              -6.393400],
    'Mo':    [np.array([3.702500, 17.235600, 12.887600, 3.742900]),
              np.array([0.277200, 1.095800, 11.004000, 61.658400]),
              4.387500],
    'Mo3+':  [np.array([21.166400, 18.201700, 11.742300, 2.309510]),
              np.array([0.014734, 1.030310, 9.536590, 26.630700]),
              -14.421000],
    'Mo5+':  [np.array([21.014900, 18.099200, 11.463200, 0.740625]),
              np.array([0.014345, 1.022380, 8.788090, 23.345200]),
              -14.316000],
    'Mo6+':  [np.array([17.887100, 11.175000, 6.578910, 0.000000]),
              np.array([1.036490, 8.480610, 0.058881, 0.000000]),
              0.344941],
    'Tc':    [np.array([19.130100, 11.094800, 4.649010, 2.712630]),
              np.array([0.864132, 8.144870, 21.570700, 86.847200]),
              5.404280],
    'Ru':    [np.array([19.267400, 12.918200, 4.863370, 1.567560]),
              np.array([0.808520, 8.434670, 24.799700, 94.292800]),
              5.378740],
    'Ru3+':  [np.array([18.563800, 13.288500, 9.326020, 3.009640]),
              np.array([0.847329, 8.371640, 0.017662, 22.887000]),
              -3.189200],
    'Ru4+':  [np.array([18.500300, 13.178700, 4.713040, 2.185350]),
              np.array([0.844582, 8.125340, 0.364950, 20.850400]),
              1.423570],
    'Rh':    [np.array([19.295700, 14.350100, 4.734250, 1.289180]),
              np.array([0.751536, 8.217580, 25.874900, 98.606200]),
              5.328000],
    'Rh3+':  [np.array([18.878500, 14.125900, 3.325150, -6.198900]),
              np.array([0.764252, 7.844380, 21.248700, -0.010360]),
              11.867800],
    'Rh4+':  [np.array([18.854500, 13.980600, 2.534640, -5.652600]),
              np.array([0.760825, 7.624360, 19.331700, -0.010200]),
              11.283500],
    'Pd':    [np.array([19.331900, 15.501700, 5.295370, 0.605844]),
              np.array([0.698655, 7.989290, 25.205200, 76.898600]),
              5.265930],
    'Pd2+':  [np.array([19.170100, 15.209600, 4.322340, 0.000000]),
              np.array([0.696219, 7.555730, 22.505700, 0.000000]),
              5.291600],
    'Pd4+':  [np.array([19.249300, 14.790000, 2.892890, -7.949200]),
              np.array([0.683839, 7.148330, 17.914400, 0.005127]),
              13.017400],
    'Ag':    [np.array([19.280800, 16.688500, 4.804500, 1.046300]),
              np.array([0.644600, 7.472600, 24.660500, 99.815600]),
              5.179000],
    'Ag1+':  [np.array([19.181200, 15.971900, 5.274750, 0.357534]),
              np.array([0.646179, 7.191230, 21.732600, 66.114700]),
              5.215720],
    'Ag2+':  [np.array([19.164300, 16.245600, 4.370900, 0.000000]),
              np.array([0.645643, 7.185440, 21.407200, 0.000000]),
              5.214040],
    'Cd':    [np.array([19.221400, 17.644400, 4.461000, 1.602900]),
              np.array([0.594600, 6.908900, 24.700800, 87.482500]),
              5.069400],
    'Cd2+':  [np.array([19.151400, 17.253500, 4.471280, 0.000000]),
              np.array([0.597922, 6.806390, 20.252100, 0.000000]),
              5.119370],
    'In':    [np.array([19.162400, 18.559600, 4.294800, 2.039600]),
              np.array([0.547600, 6.377600, 25.849900, 92.802900]),
              4.939100],
    'In3+':  [np.array([19.104500, 18.110800, 3.788970, 0.000000]),
              np.array([0.551522, 6.324700, 17.359500, 0.000000]),
              4.996350],
    'Sn':    [np.array([19.188900, 19.100500, 4.458500, 2.466300]),
              np.array([5.830300, 0.503100, 26.890900, 83.957100]),
              4.782100],
    'Sn2+':  [np.array([19.109400, 19.054800, 4.564800, 0.487000]),
              np.array([0.503600, 5.837800, 23.375200, 62.206100]),
              4.786100],
    'Sn4+':  [np.array([18.933300, 19.713100, 3.418200, 0.019300]),
              np.array([5.764000, 0.465500, 14.004900, -0.758300]),
              3.918200],
    'Sb':    [np.array([19.641800, 19.045500, 5.037100, 2.682700]),
              np.array([5.303400, 0.460700, 27.907400, 75.282500]),
              4.590900],
    'Sb3+':  [np.array([18.975500, 18.933000, 5.107890, 0.288753]),
              np.array([0.467196, 5.221260, 19.590200, 55.511300]),
              4.696260],
    'Sb5+':  [np.array([19.868500, 19.030200, 2.412530, 0.000000]),
              np.array([5.448530, 0.467973, 14.125900, 0.000000]),
              4.692630],
    'Te':    [np.array([19.964400, 19.013800, 6.144870, 2.523900]),
              np.array([4.817420, 0.420885, 28.528400, 70.840300]),
              4.352000],
    'I':     [np.array([20.147200, 18.994900, 7.513800, 2.273500]),
              np.array([4.347000, 0.381400, 27.766000, 66.877600]),
              4.071200],
    'I1-':   [np.array([20.233200, 18.997000, 7.806900, 2.886800]),
              np.array([4.357900, 0.381500, 29.525900, 84.930400]),
              4.071400],
    'Xe':    [np.array([20.293300, 19.029800, 8.976700, 1.990000]),
              np.array([3.928200, 0.344000, 26.465900, 64.265800]),
              3.711800],
    'Cs':    [np.array([20.389200, 19.106200, 10.662000, 1.495300]),
              np.array([3.569000, 0.310700, 24.387900, 213.904000]),
              3.335200],
    'Cs1+':  [np.array([20.352400, 19.127800, 10.282100, 0.961500]),
              np.array([3.552000, 0.308600, 23.712800, 59.456500]),
              3.279100],
    'Ba':    [np.array([20.336100, 19.297000, 10.888000, 2.695900]),
              np.array([3.216000, 0.275600, 20.207300, 167.202000]),
              2.773100],
    'Ba2+':  [np.array([20.180700, 19.113600, 10.905400, 0.776340]),
              np.array([3.213670, 0.283310, 20.055800, 51.746000]),
              3.029020],
    'La':    [np.array([20.578000, 19.599000, 11.372700, 3.287190]),
              np.array([2.948170, 0.244475, 18.772600, 133.124000]),
              2.146780],
    'La3+':  [np.array([20.248900, 19.376300, 11.632300, 0.336048]),
              np.array([2.920700, 0.250698, 17.821100, 54.945300]),
              2.408600],
    'Ce':    [np.array([21.167100, 19.769500, 11.851300, 3.330490]),
              np.array([2.812190, 0.226836, 17.608300, 127.113000]),
              1.862640],
    'Ce3+':  [np.array([20.803600, 19.559000, 11.936900, 0.612376]),
              np.array([2.776910, 0.231540, 16.540800, 43.169200]),
              2.090130],
    'Ce4+':  [np.array([20.323500, 19.818600, 12.123300, 0.144583]),
              np.array([2.659410, 0.218850, 15.799200, 62.235500]),
              1.591800],
    'Pr':    [np.array([22.044000, 19.669700, 12.385600, 2.824280]),
              np.array([2.773930, 0.222087, 16.766900, 143.644000]),
              2.058300],
    'Pr3+':  [np.array([21.372700, 19.749100, 12.132900, 0.975180]),
              np.array([2.645200, 0.214299, 15.323000, 36.406500]),
              1.771320],
    'Pr4+':  [np.array([20.941300, 20.053900, 12.466800, 0.296689]),
              np.array([2.544670, 0.202481, 14.813700, 45.464300]),
              1.242850],
    'Nd':    [np.array([22.684500, 19.684700, 12.774000, 2.851370]),
              np.array([2.662480, 0.210628, 15.885000, 137.903000]),
              1.984860],
    'Nd3+':  [np.array([21.961000, 19.933900, 12.120000, 1.510310]),
              np.array([2.527220, 0.199237, 14.178300, 30.871700]),
              1.475880],
    'Pm':    [np.array([23.340500, 19.609500, 13.123500, 2.875160]),
              np.array([2.562700, 0.202088, 15.100900, 132.721000]),
              2.028760],
    'Pm3+':  [np.array([22.552700, 20.110800, 12.067100, 2.074920]),
              np.array([2.417400, 0.185769, 13.127500, 27.449100]),
              1.194990],
    'Sm':    [np.array([24.004200, 19.425800, 13.439600, 2.896040]),
              np.array([2.472740, 0.196451, 14.399600, 128.007000]),
              2.209630],
    'Sm3+':  [np.array([23.150400, 20.259900, 11.920200, 2.714880]),
              np.array([2.316410, 0.174081, 12.157100, 24.824200]),
              0.954586],
    'Eu':    [np.array([24.627400, 19.088600, 13.760300, 2.922700]),
              np.array([2.387900, 0.194200, 13.754600, 123.174000]),
              2.574500],
    'Eu2+':  [np.array([24.006300, 19.950400, 11.803400, 3.872430]),
              np.array([2.277830, 0.173530, 11.609600, 26.515600]),
              1.363890],
    'Eu3+':  [np.array([23.749700, 20.374500, 11.850900, 3.265030]),
              np.array([2.222580, 0.163940, 11.311000, 22.996600]),
              0.759344],
    'Gd':    [np.array([25.070900, 19.079800, 13.851800, 3.545450]),
              np.array([2.253410, 0.181951, 12.933100, 101.398000]),
              2.419600],
    'Gd3+':  [np.array([24.346600, 20.420800, 11.870800, 3.714900]),
              np.array([2.135530, 0.155525, 10.578200, 21.702900]),
              0.645089],
    'Tb':    [np.array([25.897600, 18.218500, 14.316700, 2.953540]),
              np.array([2.242560, 0.196143, 12.664800, 115.362000]),
              3.583240],
    'Tb3+':  [np.array([24.955900, 20.327100, 12.247100, 3.773000]),
              np.array([2.056010, 0.149525, 10.049900, 21.277300]),
              0.691967],
    'Dy':    [np.array([26.507000, 17.638300, 14.559600, 2.965770]),
              np.array([2.180200, 0.202172, 12.189900, 111.874000]),
              4.297280],
    'Dy3+':  [np.array([25.539500, 20.286100, 11.981200, 4.500730]),
              np.array([1.980400, 0.143384, 9.349720, 19.581000]),
              0.689690],
    'Ho':    [np.array([26.904900, 17.294000, 14.558300, 3.638370]),
              np.array([2.070510, 0.197940, 11.440700, 92.656600]),
              4.567960],
    'Ho3+':  [np.array([26.129600, 20.099400, 11.978800, 4.936760]),
              np.array([1.910720, 0.139358, 8.800180, 18.590800]),
              0.852795],
    'Er':    [np.array([27.656300, 16.428500, 14.977900, 2.982330]),
              np.array([2.073560, 0.223545, 11.360400, 105.703000]),
              5.920460],
    'Er3+':  [np.array([26.722000, 19.774800, 12.150600, 5.173790]),
              np.array([1.846590, 0.137290, 8.362250, 17.897400]),
              1.176130],
    'Tm':    [np.array([28.181900, 15.885100, 15.154200, 2.987060]),
              np.array([2.028590, 0.238849, 10.997500, 102.961000]),
              6.756210],
    'Tm3+':  [np.array([27.308300, 19.332000, 12.333900, 5.383480]),
              np.array([1.787110, 0.136974, 7.967780, 17.292200]),
              1.639290],
    'Yb':    [np.array([28.664100, 15.434500, 15.308700, 2.989630]),
              np.array([1.988900, 0.257119, 10.664700, 100.417000]),
              7.566720],
    'Yb2+':  [np.array([28.120900, 17.681700, 13.333500, 5.146570]),
              np.array([1.785030, 0.159970, 8.183040, 20.390000]),
              3.709830],
    'Yb3+':  [np.array([27.891700, 18.761400, 12.607200, 5.476470]),
              np.array([1.732720, 0.138790, 7.644120, 16.815300]),
              2.260010],
    'Lu':    [np.array([28.947600, 15.220800, 15.100000, 3.716010]),
              np.array([1.901820, 9.985190, 0.261033, 84.329800]),
              7.976280],
    'Lu3+':  [np.array([28.462800, 18.121000, 12.842900, 5.594150]),
              np.array([1.682160, 0.142292, 7.337270, 16.353500]),
              2.975730],
    'Hf':    [np.array([29.144000, 15.172600, 14.758600, 4.300130]),
              np.array([1.832620, 9.599900, 0.275116, 72.029000]),
              8.581540],
    'Hf4+':  [np.array([28.813100, 18.460100, 12.728500, 5.599270]),
              np.array([1.591360, 0.128903, 6.762320, 14.036600]),
              2.396990],
    'Ta':    [np.array([29.202400, 15.229300, 14.513500, 4.764920]),
              np.array([1.773330, 9.370460, 0.295977, 63.364400]),
              9.243540],
    'Ta5+':  [np.array([29.158700, 18.840700, 12.826800, 5.386950]),
              np.array([1.507110, 0.116741, 6.315240, 12.424400]),
              1.785550],
    'W':     [np.array([29.081800, 15.430000, 14.432700, 5.119820]),
              np.array([1.720290, 9.225900, 0.321703, 57.056000]),
              9.887500],
    'W6+':   [np.array([29.493600, 19.376300, 13.054400, 5.064120]),
              np.array([1.427550, 0.104621, 5.936670, 11.197200]),
              1.010740],
    'Re':    [np.array([28.762100, 15.718900, 14.556400, 5.441740]),
              np.array([1.671910, 9.092270, 0.350500, 52.086100]),
              10.472000],
    'Os':    [np.array([28.189400, 16.155000, 14.930500, 5.675890]),
              np.array([1.629030, 8.979480, 0.382661, 48.164700]),
              11.000500],
    'Os4+':  [np.array([30.419000, 15.263700, 14.745800, 5.067950]),
              np.array([1.371130, 6.847060, 0.165191, 18.003000]),
              6.498040],
    'Ir':    [np.array([27.304900, 16.729600, 15.611500, 5.833770]),
              np.array([1.592790, 8.865530, 0.417916, 45.001100]),
              11.472200],
    'Ir3+':  [np.array([30.415600, 15.862000, 13.614500, 5.820080]),
              np.array([1.343230, 7.109090, 0.204633, 20.325400]),
              8.279030],
    'Ir4+':  [np.array([30.705800, 15.551200, 14.232600, 5.536720]),
              np.array([1.309230, 6.719830, 0.167252, 17.491100]),
              6.968240],
    'Pt':    [np.array([27.005900, 17.763900, 15.713100, 5.783700]),
              np.array([1.512930, 8.811740, 0.424593, 38.610300]),
              11.688300],
    'Pt2+':  [np.array([29.842900, 16.722400, 13.215300, 6.352340]),
              np.array([1.329270, 7.389790, 0.263297, 22.942600]),
              9.853290],
    'Pt4+':  [np.array([30.961200, 15.982900, 13.734800, 5.920340]),
              np.array([1.248130, 6.608340, 0.168640, 16.939200]),
              7.395340],
    'Au':    [np.array([16.881900, 18.591300, 25.558200, 5.860000]),
              np.array([0.461100, 8.621600, 1.482600, 36.395600]),
              12.065800],
    'Au1+':  [np.array([28.010900, 17.820400, 14.335900, 6.580770]),
              np.array([1.353210, 7.739500, 0.356752, 26.404300]),
              11.229900],
    'Au3+':  [np.array([30.688600, 16.902900, 12.780100, 6.523540]),
              np.array([1.219900, 6.828720, 0.212867, 18.659000]),
              9.096800],
    'Hg':    [np.array([20.680900, 19.041700, 21.657500, 5.967600]),
              np.array([0.545000, 8.448400, 1.572900, 38.324600]),
              12.608900],
    'Hg1+':  [np.array([25.085300, 18.497300, 16.888300, 6.482160]),
              np.array([1.395070, 7.651050, 0.443378, 28.226200]),
              12.020500],
    'Hg2+':  [np.array([29.564100, 18.060000, 12.837400, 6.899120]),
              np.array([1.211520, 7.056390, 0.284738, 20.748200]),
              10.626800],
    'Tl':    [np.array([27.544600, 19.158400, 15.538000, 5.525930]),
              np.array([0.655150, 8.707510, 1.963470, 45.814900]),
              13.174600],
    'Tl1+':  [np.array([21.398500, 20.472300, 18.747800, 6.828470]),
              np.array([1.471100, 0.517394, 7.434630, 28.848200]),
              12.525800],
    'Tl3+':  [np.array([30.869500, 18.348100, 11.932800, 7.005740]),
              np.array([1.100800, 6.538520, 0.219074, 17.211400]),
              9.802700],
    'Pb':    [np.array([31.061700, 13.063700, 18.442000, 5.969600]),
              np.array([0.690200, 2.357600, 8.618000, 47.257900]),
              13.411800],
    'Pb2+':  [np.array([21.788600, 19.568200, 19.140600, 7.011070]),
              np.array([1.336600, 0.488383, 6.772700, 23.813200]),
              12.473400],
    'Pb4+':  [np.array([32.124400, 18.800300, 12.017500, 6.968860]),
              np.array([1.005660, 6.109260, 0.147041, 14.714000]),
              8.084280],
    'Bi':    [np.array([33.368900, 12.951000, 16.587700, 6.469200]),
              np.array([0.704000, 2.923800, 8.793700, 48.009300]),
              13.578200],
    'Bi3+':  [np.array([21.805300, 19.502600, 19.105300, 7.102950]),
              np.array([1.235600, 6.241490, 0.469999, 20.318500]),
              12.471100],
    'Bi5+':  [np.array([33.536400, 25.094600, 19.249700, 6.915550]),
              np.array([0.916540, 0.390420, 5.714140, 12.828500]),
              -6.799400],
    'Po':    [np.array([34.672600, 15.473300, 13.113800, 7.025880]),
              np.array([0.700999, 3.550780, 9.556420, 47.004500]),
              13.677000],
    'At':    [np.array([35.316300, 19.021100, 9.498870, 7.425180]),
              np.array([0.685870, 3.974580, 11.382400, 45.471500]),
              13.710800],
    'Rn':    [np.array([35.563100, 21.281600, 8.003700, 7.443300]),
              np.array([0.663100, 4.069100, 14.042200, 44.247300]),
              13.690500],
    'Fr':    [np.array([35.929900, 23.054700, 12.143900, 2.112530]),
              np.array([0.646453, 4.176190, 23.105200, 150.645000]),
              13.724700],
    'Ra':    [np.array([35.763000, 22.906400, 12.473900, 3.210970]),
              np.array([0.616341, 3.871350, 19.988700, 142.325000]),
              13.621100],
    'Ra2+':  [np.array([35.215000, 21.670000, 7.913420, 7.650780]),
              np.array([0.604909, 3.576700, 12.601000, 29.843600]),
              13.543100],
    'Ac':    [np.array([35.659700, 23.103200, 12.597700, 4.086550]),
              np.array([0.589092, 3.651550, 18.599000, 117.020000]),
              13.526600],
    'Ac3+':  [np.array([35.173600, 22.111200, 8.192160, 7.055450]),
              np.array([0.579689, 3.414370, 12.918700, 25.944300]),
              13.463700],
    'Th':    [np.array([35.564500, 23.421900, 12.747300, 4.807030]),
              np.array([0.563359, 3.462040, 17.830900, 99.172200]),
              13.431400],
    'Th4+':  [np.array([35.100700, 22.441800, 9.785540, 5.294440]),
              np.array([0.555054, 3.244980, 13.466100, 23.953300]),
              13.376000],
    'Pa':    [np.array([35.884700, 23.294800, 14.189100, 4.172870]),
              np.array([0.547751, 3.415190, 16.923500, 105.251000]),
              13.428700],
    'U':     [np.array([36.022800, 23.412800, 14.949100, 4.188000]),
              np.array([0.529300, 3.325300, 16.092700, 100.613000]),
              13.396600],
    'U3+':   [np.array([35.574700, 22.525900, 12.216500, 5.370730]),
              np.array([0.520480, 3.122930, 12.714800, 26.339400]),
              13.309200],
    'U4+':   [np.array([35.371500, 22.532600, 12.029100, 4.798400]),
              np.array([0.516598, 3.050530, 12.572300, 23.458200]),
              13.267100],
    'U6+':   [np.array([34.850900, 22.758400, 14.009900, 1.214570]),
              np.array([0.507079, 2.890300, 13.176700, 25.201700]),
              13.166500],
    'Np':    [np.array([36.187400, 23.596400, 15.640200, 4.185500]),
              np.array([0.511929, 3.253960, 15.362200, 97.490800]),
              13.357300],
    'Np3+':  [np.array([35.707400, 22.613000, 12.989800, 5.432270]),
              np.array([0.502322, 3.038070, 12.144900, 25.492800]),
              13.254400],
    'Np4+':  [np.array([35.510300, 22.578700, 12.776600, 4.921590]),
              np.array([0.498626, 2.966270, 11.948400, 22.750200]),
              13.211600],
    'Np6+':  [np.array([35.013600, 22.728600, 14.388400, 1.756690]),
              np.array([0.489810, 2.810990, 12.330000, 22.658100]),
              13.113000],
    'Pu':    [np.array([36.525400, 23.808300, 16.770700, 3.479470]),
              np.array([0.499384, 3.263710, 14.945500, 105.980000]),
              13.381200],
    'Pu3+':  [np.array([35.840000, 22.716900, 13.580700, 5.660160]),
              np.array([0.484938, 2.961180, 11.533100, 24.399200]),
              13.199100],
    'Pu4+':  [np.array([35.649300, 22.646000, 13.359500, 5.188310]),
              np.array([0.481422, 2.890200, 11.316000, 21.830100]),
              13.155500],
    'Pu6+':  [np.array([35.173600, 22.718100, 14.763500, 2.286780]),
              np.array([0.473204, 2.738480, 11.553000, 20.930300]),
              13.058200],
    'Am':    [np.array([36.670600, 24.099200, 17.341500, 3.493310]),
              np.array([0.483629, 3.206470, 14.313600, 102.273000]),
              13.359200],
    'Cm':    [np.array([36.648800, 24.409600, 17.399000, 4.216650]),
              np.array([0.465154, 3.089970, 13.434600, 88.483400]),
              13.288700],
    'Bk':    [np.array([36.788100, 24.773600, 17.891900, 4.232840]),
              np.array([0.451018, 3.046190, 12.894600, 86.003000]),
              13.275400],
    'Cf':    [np.array([36.918500, 25.199500, 18.331700, 4.243910]),
              np.array([0.437533, 3.007750, 12.404400, 83.788100]),
              13.267400]
}

def calc_f0j(cell_mat_m, element_symbols, positions, index_vec_h, symm_mats_vecs, gpaw_dict=None, restart=None, save=None, explicit_core=False):
    cell_mat_f = np.linalg.inv(cell_mat_m)
    vec_S_norm = np.linalg.norm(np.einsum('xy, hy -> hx', cell_mat_f.T, index_vec_h), axis=-1)
    n_symm = symm_mats_vecs[0].shape[0]
    f0j_a = np.array([element_parameters[symbol][0] for symbol in element_symbols])
    f0j_b = np.array([element_parameters[symbol][1] for symbol in element_symbols])
    f0j_c = np.array([element_parameters[symbol][2] for symbol in element_symbols])

    fjs = np.einsum('k, zi, izh -> kzh', np.ones(n_symm), f0j_a,  np.exp(-1*np.einsum('h, zi -> izh', (vec_S_norm / 2)**2, f0j_b))) + f0j_c[None, :, None]
    return fjs, None

def calculate_f0j_core(cell_mat_m, element_symbols, positions, index_vec_h, symm_mats_vecs):
    raise NotImplementedError('Separate core calculation is non-sensical in IAM Mode')