# SuperOdom-M
Minmal ICP case of SuperOdom for degenercy detetcion, only depending on Ceres and Eigen.



```c++
#include "superloc_icp_integration.h"

// 运行SuperLoc ICP
SuperLocICPIntegration::SuperLocResult superloc_result;
auto start_total = std::chrono::high_resolution_clock::now();

// 调用SuperLoc ICP，使用默认的plane_resolution = 0.1
bool success = SuperLocICPIntegration::runSuperLocICP(
        source_cloud_,
        target_cloud_,
        initial_matrix,
        config_.max_iterations,
        config_.search_radius,
        superloc_result,
        result.final_transform,
        result.iteration_data,  // 直接使用result的iteration_data
        0.1  // plane_resolution参数，对应原始SuperLoc的localMap.planeRes_
);

auto end_total = std::chrono::high_resolution_clock::now();

```



Related package:  https://github.com/superxslam/SuperOdom

 
