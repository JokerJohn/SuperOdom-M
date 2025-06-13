//
// Created by xchu on 12/6/2025.
//

#ifndef CLOUD_MAP_EVALUATION_UTILS_H
#define CLOUD_MAP_EVALUATION_UTILS_H

#include <pcl/io/pcd_io.h>
#include <pcl/common/distances.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <fstream>
#include <chrono>
#include <map>
#include <vector>
#include <string>
#include <filesystem>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/angles.h> // for pcl::rad2deg, pcl::deg2rad

// Define PointT if not defined elsewhere
// typedef pcl::PointXYZI PointT; // Example: Use PointXYZI
// Or use the one from the original code if it's different and accessible
using PointT = pcl::PointXYZI; // Using PointXYZ as a common default
using namespace Eigen;
namespace fs = std::filesystem;

namespace ICPRunner {
    // Simple Timer (same as before)
    class TicToc {
    public:
        TicToc() { tic(); }

        void tic() { start = std::chrono::system_clock::now(); }

        double toc() {
            end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            return elapsed_seconds.count() * 1000; // return ms
        }

    private:
        std::chrono::time_point <std::chrono::system_clock> start, end;
    };

    // Pose Representation (same as before)
    struct Pose6D {
        double roll = 0.0, pitch = 0.0, yaw = 0.0;
        double x = 0.0, y = 0.0, z = 0.0;

        Pose6D operator+(const Pose6D &other) const {
            Pose6D result;
            result.roll = roll + other.roll;
            result.pitch = pitch + other.pitch;
            result.yaw = yaw + other.yaw;
            result.x = x + other.x;
            result.y = y + other.y;
            result.z = z + other.z;
            return result;
        }
    };

    // Tunable Parameters - will be updated from config
    struct ICPParameters {
        double DEGENERACY_THRES_COND = 10.0;
        double DEGENERACY_THRES_EIG = 120.0;
        double KAPPA_TARGET = 1.0;
        double PCG_TOLERANCE = 1e-6;
        int PCG_MAX_ITER = 10;
        double ADAPTIVE_REG_ALPHA = 10.0;
        double STD_REG_GAMMA = 0.01;
        double LOAM_EIGEN_THRESH = 120.0;
        double TSVD_SINGULAR_THRESH = 120.0;
    };

    // Degeneracy Detection Methods
    enum class DegeneracyDetectionMethod {
        NONE_DETE, SCHUR_CONDITION_NUMBER, FULL_EVD_MIN_EIGENVALUE,
        EVD_SUB_CONDITION, FULL_SVD_CONDITION, O3D, SUPERLOC
    };

// Degeneracy Handling Methods
    enum class DegeneracyHandlingMethod {
        NONE_HAND, STANDARD_REGULARIZATION, ADAPTIVE_REGULARIZATION,
        PRECONDITIONED_CG, SOLUTION_REMAPPING, TRUNCATED_SVD, O3D, SUPERLOC
    };


    enum class ICPEngine {
        CUSTOM_EULER,    // 自定义欧拉角实现
        CUSTOM_SO3,      // 自定义SO(3)实现
        OPEN3D           // Open3D实现
    };


    // Configuration structure
    struct Config {
        // Test configuration
        int num_runs = 1;
        bool save_pcd = true;
        bool save_error_pcd = true;
        bool visualize = false;

        // File paths
        std::string folder_path;
        std::string source_pcd;
        std::string target_pcd;
        std::string output_folder;

        // ICP parameters
        double search_radius = 1.0;
        int max_iterations = 30;
        double error_threshold = 0.05; // for visualization

        // Initial noise
        Pose6D initial_noise;

        // Ground truth
        bool use_identity_gt = true;
        Pose6D ground_truth;

        // ICP parameter values
        ICPParameters icp_params;

        ICPEngine icp_engine = ICPEngine::CUSTOM_EULER;

        // Test methods
        std::map <std::string, std::pair<std::string, std::string>> test_methods;

        // Engine selection
        bool use_custom_engine = true;

        // Use SO(3) parameterization
        bool use_so3_parameterization = true;
    };

    // --- CSV Logging Data Structure ---
    struct IterationLogData {
        int iter_count = 0;
        int effective_points = 0;
        double rmse = 0.0;
        double fitness = 0.0;
        int corr_num = 0;

        double iter_time_ms = 0.0;
        double cond_schur_rot = NAN;
        double cond_schur_trans = NAN;
        double cond_diag_rot = NAN;
        double cond_diag_trans = NAN;
        double cond_full_evd_sub_rot = NAN;
        double cond_full_evd_sub_trans = NAN;
        double cond_full_svd = NAN;
        double cond_full = NAN;  // 直接从H = J^T * J计算的条件数
        Eigen::Vector3d lambda_schur_rot = Eigen::Vector3d::Constant(NAN);
        Eigen::Vector3d lambda_schur_trans = Eigen::Vector3d::Constant(NAN);
        Eigen::Matrix<double, 6, 1> eigenvalues_full = Eigen::Matrix<double, 6, 1>::Constant(NAN);
        Eigen::Matrix<double, 6, 1> singular_values_full = Eigen::Matrix<double, 6, 1>::Constant(NAN);
        Eigen::Matrix<double, 6, 1> update_dx = Eigen::Matrix<double, 6, 1>::Constant(NAN);
        bool is_degenerate = false;
        std::vector<bool> degenerate_mask = std::vector<bool>(6, false);

        Eigen::Matrix4d transform_matrix = Eigen::Matrix4d::Identity(); // 添加变换矩阵
        double trans_error_vs_gt = 0.0;  // 相对于真值的平移误差
        double rot_error_vs_gt = 0.0;    // 相对于真值的旋转误差

        // 新增：对角块特征值
        Eigen::Vector3d lambda_diag_rot = Eigen::Vector3d::Constant(NAN);
        Eigen::Vector3d lambda_diag_trans = Eigen::Vector3d::Constant(NAN);

        // 新增：预处理矩阵
        Eigen::Matrix<double, 6, 6> P_preconditioner = Eigen::Matrix<double, 6, 6>::Identity();
        Eigen::Matrix<double, 6, 6> W_adaptive = Eigen::Matrix<double, 6, 6>::Zero();


        // 新增：对于Schur+PCG方法的特殊信息
        Eigen::Matrix3d aligned_V_rot = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d aligned_V_trans = Eigen::Matrix3d::Identity();
        std::vector<int> rot_indices = {0, 1, 2};
        std::vector<int> trans_indices = {0, 1, 2};

        // 新增：梯度信息（-J^T * r）
        Eigen::Matrix<double, 6, 1> gradient = Eigen::Matrix<double, 6, 1>::Constant(NAN);

        // 新增：目标函数值（0.5 * r^T * r）
        double objective_value = NAN;

    };


    // Test result structure
    struct TestResult {
        std::string method_name;
        bool converged = false;
        int iterations = 0;
        double time_ms = 0.0;
        double trans_error_m = 0.0;
        double rot_error_deg = 0.0;
        double final_rmse = 0.0;
        double final_fitness = 0.0;
        double p2p_rmse = 0.0;
        double p2p_fitness = 0.0;
        double chamfer_distance = 0.0;
        int corr_num = 0;
        Eigen::Matrix4d final_transform = Eigen::Matrix4d::Identity();

        // Degeneracy analysis (for single run)
        std::vector<double> condition_numbers;
        std::vector<double> eigenvalues;
        std::vector<double> singular_values;
        std::vector<bool> degenerate_mask;

        // Iteration history for plotting
        std::vector<double> iter_rmse_history;
        std::vector<double> iter_fitness_history;
        std::vector<int> iter_corr_num_history;
        std::vector<double> iter_trans_error_history;
        std::vector<double> iter_rot_error_history;

        // 新增：保存每次迭代的完整数据
        std::vector <IterationLogData> iteration_data;

        // 新增：保存每次迭代的变换矩阵
        std::vector <Eigen::Matrix4d> iter_transform_history;

        // SuperLoc特有的数据
        struct SuperLocData {
            bool has_data = false;
            double uncertainty_x = 0.0;
            double uncertainty_y = 0.0;
            double uncertainty_z = 0.0;
            double uncertainty_roll = 0.0;
            double uncertainty_pitch = 0.0;
            double uncertainty_yaw = 0.0;
            double cond_full = 0.0;
            double cond_rot = 0.0;
            double cond_trans = 0.0;
            bool is_degenerate = false;
            Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Identity();
            std::array<int, 9> feature_histogram = {0, 0, 0, 0, 0, 0, 0, 0, 0};  // 新增：特征可观测性直方图
        } superloc_data;
    };

// Statistics for multiple runs
    struct MethodStatistics {
        std::string method_name;
        int total_runs = 0;
        int converged_runs = 0;
        int corr_num = 0;

        // Mean values
        double mean_trans_error = 0.0;
        double mean_rot_error = 0.0;
        double mean_time_ms = 0.0;
        double mean_iterations = 0.0;
        double mean_rmse = 0.0;
        double mean_fitness = 0.0;
        double mean_p2p_rmse = 0.0;
        double mean_p2p_fitness = 0.0;
        double mean_chamfer = 0.0;

        // Standard deviations
        double std_trans_error = 0.0;
        double std_rot_error = 0.0;
        double std_time_ms = 0.0;

        // Min/Max values
        double min_trans_error = std::numeric_limits<double>::max();
        double max_trans_error = 0.0;
        double min_rot_error = std::numeric_limits<double>::max();
        double max_rot_error = 0.0;

        // Success rate
        double success_rate = 0.0;
    };


    // --- ICP State Context ---
    // Holds variables that persist across calls or are needed internally by ICP
    struct ICPContext {
        // Pointers to clouds used internally
        pcl::PointCloud<PointT>::Ptr laserCloudEffective;
        pcl::PointCloud<PointT>::Ptr coeffSel; // Stores weighted normal (xyz) and residual (intensity)

        // Internal state vectors (resized based on input cloud)
        std::vector <PointT> laserCloudOriSurfVec;
        std::vector <PointT> coeffSelSurfVec;
        std::vector <uint8_t> laserCloudOriSurfFlag; // Use uint8_t for bool efficiency

        // KdTree for the target map
        pcl::KdTreeFLANN<PointT>::Ptr kdtreeSurfFromMap;

        // Result Storage
        Eigen::Matrix<double, 6, 6> icp_cov; // Final computed covariance
        std::vector <IterationLogData> iteration_log_data_; // Log data per iteration
        double total_icp_time_ms_ = 0.0;
        Pose6D final_pose_; // The final optimized pose
        bool final_convergence_flag_ = false;
        int final_iterations_ = 0; // Store the number of iterations performed

        // Constructor to initialize pointers and default covariance
        ICPContext() :
                laserCloudEffective(new pcl::PointCloud<PointT>()),
                coeffSel(new pcl::PointCloud<PointT>()),
                kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointT>()),
                icp_cov(Eigen::Matrix<double, 6, 6>::Identity()) {
            icp_cov *= 1e6; // Default high covariance
        }

        // Prevent copying (pointers would be shallow copied)
        ICPContext(const ICPContext &) = delete;

        ICPContext &operator=(const ICPContext &) = delete;

        // Setup method (optional, can be done in runSingleICPTest)
        void setTargetCloud(pcl::PointCloud<PointT>::Ptr target_cloud_ptr) {
            if (!target_cloud_ptr || target_cloud_ptr->empty()) {
                std::cerr << "[ICPContext::setTargetCloud] Error: Target cloud is null or empty." << std::endl;
                return;
            }
            kdtreeSurfFromMap->setInputCloud(target_cloud_ptr);
            std::cout << "[ICPContext::setTargetCloud] KdTree built for target cloud with "
                      << target_cloud_ptr->size() << " points." << std::endl;
        }
    };

    struct DegeneracyAnalysisResult {
        bool isDegenerate = false;
        std::vector<bool> degenerate_mask;
        double cond_schur_rot = NAN, cond_schur_trans = NAN;
        double cond_diag_rot = NAN, cond_diag_trans = NAN;
        double cond_full = NAN;
        double cond_full_sub_rot = NAN, cond_full_sub_trans = NAN;
        Eigen::Matrix<double, 6, 1> eigenvalues_full;
        Eigen::Matrix<double, 6, 1> singular_values;
        Eigen::Vector3d lambda_schur_rot = Eigen::Vector3d::Constant(NAN);
        Eigen::Vector3d lambda_schur_trans = Eigen::Vector3d::Constant(NAN);
        Eigen::Vector3d lambda_sub_rot = Eigen::Vector3d::Constant(NAN);
        Eigen::Vector3d lambda_sub_trans = Eigen::Vector3d::Constant(NAN);
        Eigen::Matrix3d aligned_V_rot = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d aligned_V_trans = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d schur_V_rot = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d schur_V_trans = Eigen::Matrix3d::Identity();
        std::vector<int> rot_indices = {0, 1, 2};
        std::vector<int> trans_indices = {0, 1, 2};
        Eigen::Matrix<double, 6, 6> W_adaptive = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 6> P_preconditioner = Eigen::Matrix<double, 6, 6>::Identity();
    };

}

#endif //CLOUD_MAP_EVALUATION_UTILS_H
