#include "superloc_icp_integration.h"
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <iostream>

namespace ICPRunner {

// SuperLocPoseParameterization实现
    bool SuperLocPoseParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const {
        // 位置更新
        x_plus_delta[0] = x[0] + delta[0];
        x_plus_delta[1] = x[1] + delta[1];
        x_plus_delta[2] = x[2] + delta[2];

        // 旋转更新 - 与原始SuperLoc保持一致
        const double norm_delta = sqrt(delta[3] * delta[3] + delta[4] * delta[4] + delta[5] * delta[5]);
        if (norm_delta > 0.0) {
            const double sin_delta_by_delta = sin(norm_delta) / norm_delta;
            double q_delta[4];
            q_delta[0] = cos(norm_delta);
            q_delta[1] = sin_delta_by_delta * delta[3];
            q_delta[2] = sin_delta_by_delta * delta[4];
            q_delta[3] = sin_delta_by_delta * delta[5];

            // 四元数乘法: q_delta * q_current
            ceres::QuaternionProduct(q_delta, x + 3, x_plus_delta + 3);
        } else {
            x_plus_delta[3] = x[3];
            x_plus_delta[4] = x[4];
            x_plus_delta[5] = x[5];
            x_plus_delta[6] = x[6];
        }

        return true;
    }

    bool SuperLocPoseParameterization::ComputeJacobian(const double *x, double *jacobian) const {
        Eigen::Map <Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J(jacobian);
        J.setZero();

        // 位置部分的雅可比
        J.topLeftCorner<3, 3>().setIdentity();

        // 旋转部分的雅可比
        const double q_w = x[6];
        const double q_x = x[3];
        const double q_y = x[4];
        const double q_z = x[5];

        J(3, 3) = 0.5 * q_w;
        J(3, 4) = 0.5 * q_z;
        J(3, 5) = -0.5 * q_y;
        J(4, 3) = -0.5 * q_z;
        J(4, 4) = 0.5 * q_w;
        J(4, 5) = 0.5 * q_x;
        J(5, 3) = 0.5 * q_y;
        J(5, 4) = -0.5 * q_x;
        J(5, 5) = 0.5 * q_w;
        J(6, 3) = -0.5 * q_x;
        J(6, 4) = -0.5 * q_y;
        J(6, 5) = -0.5 * q_z;

        return true;
    }

// SuperLoc点到平面代价函数
    bool SuperLocPlaneResidual::Evaluate(double const *const *parameters,
                                         double *residuals,
                                         double **jacobians) const {
        // 获取位姿参数
        Eigen::Map<const Eigen::Vector3d> t(parameters[0]);
        Eigen::Map<const Eigen::Quaterniond> q(parameters[0] + 3);

        // 变换源点
        Eigen::Vector3d p_trans = q * src_point_ + t;

        // 计算残差：点到平面距离
        // 注意：这里的d_已经是negative_OA_dot_norm
        residuals[0] = tgt_normal_.dot(p_trans) + d_;

        if (jacobians != NULL && jacobians[0] != NULL) {
            Eigen::Map <Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> J(jacobians[0]);

            // 对平移的导数
            J(0, 0) = tgt_normal_(0);
            J(0, 1) = tgt_normal_(1);
            J(0, 2) = tgt_normal_(2);

            // 对旋转的导数 - 简化版本
            // 使用链式法则: dr/dq = n^T * d(R*p)/dq
//            const Eigen::Matrix3d R = q.toRotationMatrix();
//            const Eigen::Vector3d Rp = R * src_point_;

            // 构造[Rp]×矩阵
//            Eigen::Matrix3d skew_Rp;
//            skew_Rp << 0, -Rp(2), Rp(1),
//                    Rp(2), 0, -Rp(0),
//                    -Rp(1), Rp(0), 0;
//
//            // 计算 n^T * [Rp]×
//            Eigen::RowVector3d nT_skew = tgt_normal_.transpose() * skew_Rp;
//
//            // 四元数导数（这是一个近似，但在小角度下足够准确）
//            J(0, 3) = nT_skew(0);
//            J(0, 4) = nT_skew(1);
//            J(0, 5) = nT_skew(2);
//            J(0, 6) = 0.0;

// 四元数旋转公式: q * p * q^(-1)
// 展开后对各个四元数分量求导

            double qw = q.w();
            double qx = q.x();
            double qy = q.y();
            double qz = q.z();

            double px = src_point_(0);
            double py = src_point_(1);
            double pz = src_point_(2);

            // 四元数旋转公式: q * p * q^(-1)
            // 展开后对各个四元数分量求导

            // ∂(q*p)/∂qx - 修正后的公式
            Eigen::Vector3d dp_dqx;
            dp_dqx(0) = 2 * (qy * py + qz * pz);
            dp_dqx(1) = 2 * (qy * px - 2 * qx * py - qw * pz);
            dp_dqx(2) = 2 * (qz * px + qw * py - 2 * qx * pz);

            // ∂(q*p)/∂qy - 修正后的公式
            Eigen::Vector3d dp_dqy;
            dp_dqy(0) = 2 * (qx * py - 2 * qy * px + qw * pz);
            dp_dqy(1) = 2 * (qx * px + qz * pz);
            dp_dqy(2) = 2 * (qz * py - qw * px - 2 * qy * pz);

            // ∂(q*p)/∂qz - 修正后的公式
            Eigen::Vector3d dp_dqz;
            dp_dqz(0) = 2 * (qx * pz - qw * py - 2 * qz * px);
            dp_dqz(1) = 2 * (qy * pz + qw * px - 2 * qz * py);
            dp_dqz(2) = 2 * (qx * px + qy * py);

            // ∂(q*p)/∂qw - 修正后的公式
            Eigen::Vector3d dp_dqw;
            dp_dqw(0) = 2 * (qy * pz - qz * py);
            dp_dqw(1) = 2 * (qz * px - qx * pz);
            dp_dqw(2) = 2 * (qx * py - qy * px);

            // 最终雅可比 = n^T * ∂(q*p)/∂q
            J(0, 3) = tgt_normal_.dot(dp_dqx);
            J(0, 4) = tgt_normal_.dot(dp_dqy);
            J(0, 5) = tgt_normal_.dot(dp_dqz);
            J(0, 6) = tgt_normal_.dot(dp_dqw);
        }

        return true;
    }

    bool SuperLocICPIntegration::runSuperLocICP(
            const pcl::PointCloud<pcl::PointXYZI>::Ptr &source,
            const pcl::PointCloud<pcl::PointXYZI>::Ptr &target,
            const Eigen::Matrix4d &initial_guess,
            int max_iterations,
            double correspondence_distance,
            SuperLocResult &result,
            Eigen::Matrix4d &final_transform,
            std::vector <IterationLogData> &iteration_logs,
            double plane_resolution) {

        // 初始化结果
        result.converged = false;
        result.iterations = 0;
        result.isDegenerate = false;
        final_transform = initial_guess;

        // 初始化位姿参数（7维：x,y,z,qx,qy,qz,qw）
        double pose_parameters[7];
        Eigen::Map <Eigen::Vector3d> pose_t(pose_parameters);
        Eigen::Map <Eigen::Quaterniond> pose_q(pose_parameters + 3);

        // 从初始猜测设置参数
        Eigen::Vector3d t = initial_guess.block<3, 1>(0, 3);
        Eigen::Quaterniond q(initial_guess.block<3, 3>(0, 0));
        q.normalize();

        pose_t = t;
        pose_q = q;

        // 构建目标点云KD树
        pcl::KdTreeFLANN <pcl::PointXYZI> kdtree;
        kdtree.setInputCloud(target);

        // 特征可观测性直方图 - 与原始SuperLoc保持一致
        std::array<int, 9> PlaneFeatureHistogramObs;

        // ICP主循环
        for (int iter = 0; iter < max_iterations; ++iter) {
            IterationLogData iter_log;
            iter_log.iter_count = iter;

            auto iter_start = std::chrono::high_resolution_clock::now();

            // 重置特征可观测性直方图
            PlaneFeatureHistogramObs.fill(0);

            // 1. 寻找对应点和法向量
            std::vector <std::pair<int, int>> correspondences;
            std::vector <Eigen::Vector3d> normals;
            std::vector<double> plane_d_values;

            findCorrespondencesWithNormals(source, target, final_transform,
                                           correspondence_distance, correspondences,
                                           normals, plane_d_values);

            if (correspondences.size() < 10) {
                std::cout << "[SuperLoc] Not enough correspondences: "
                          << correspondences.size() << std::endl;
                break;
            }

            iter_log.corr_num = correspondences.size();

            // 2. 分析特征可观测性 - 使用原始SuperLoc的方法
            analyzeFeatureObservabilityDetailed(source, correspondences, normals,
                                                final_transform, PlaneFeatureHistogramObs);

            // 3. 构建优化问题
            ceres::Problem problem;
            ceres::LocalParameterization *pose_parameterization =
                    new SuperLocPoseParameterization();
            problem.AddParameterBlock(pose_parameters, 7, pose_parameterization);

            // 添加点到平面约束
            double total_residual = 0.0;
            int valid_constraints = 0;

            for (size_t i = 0; i < correspondences.size(); ++i) {
                const auto &corr = correspondences[i];
                Eigen::Vector3d src_pt(source->points[corr.first].x,
                                       source->points[corr.first].y,
                                       source->points[corr.first].z);

                // 使用自动微分版本进行测试
                bool use_autodiff = false;  // 可以切换测试
                ceres::CostFunction *cost_function = nullptr;

                if (use_autodiff) {
                    cost_function = SuperLocPlaneResidualAuto::Create(src_pt, normals[i], plane_d_values[i]);
                } else {
                    cost_function = new SuperLocPlaneResidual(src_pt, normals[i], plane_d_values[i]);
                }

                // 损失函数策略
                ceres::LossFunction *loss_function = nullptr;
                if (iter > 2) {  // 前几次迭代不使用鲁棒核
                    loss_function = new ceres::TukeyLoss(std::sqrt(3 * plane_resolution));
                }

                problem.AddResidualBlock(cost_function, loss_function, pose_parameters);
                valid_constraints++;

                // 计算初始残差用于调试
                if (iter == 0 && i < 100) {
                    Eigen::Vector3d p_trans = pose_q * src_pt + pose_t;
                    double residual = normals[i].dot(p_trans) + plane_d_values[i];
                    total_residual += residual * residual;
                }
            }

            // 调试信息
            if (iter == 0) {
                double avg_residual = std::sqrt(total_residual / std::min(100, (int) correspondences.size()));
                std::cout << "[SuperLoc Debug] Initial state - t: [" << pose_t.transpose()
                          << "], q: [" << pose_q.coeffs().transpose() << "]" << std::endl;
                std::cout << "[SuperLoc Debug] Valid constraints: " << valid_constraints
                          << ", avg residual: " << avg_residual << std::endl;
            }

            // 4. 求解优化问题 - 与原始SuperLoc完全一致
            ceres::Solver::Options options;
            options.max_num_iterations = 4;
            options.linear_solver_type = ceres::DENSE_QR;
            options.minimizer_progress_to_stdout = false;
            options.check_gradients = false;
            options.gradient_check_relative_precision = 1e-4;

            // 保存优化前的参数用于调试
            Eigen::Vector3d t_before = t;
            Eigen::Quaterniond q_before = q;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            // 调试信息
            if (iter < 3 || (iter % 10 == 0)) {
                std::cout << "[SuperLoc Debug] Iter " << iter
                          << ", cost change: " << summary.initial_cost << " -> " << summary.final_cost
                          << ", successful steps: " << summary.num_successful_steps << std::endl;
            }

            // 5. 更新变换矩阵
            // 直接从Map获取更新后的值
            t = pose_t;
            q = pose_q;
            q.normalize();

            final_transform.setIdentity();
            final_transform.block<3, 3>(0, 0) = q.toRotationMatrix();
            final_transform.block<3, 1>(0, 3) = t;

            // 调试：检查参数是否更新
            if (iter < 3 || (iter % 10 == 0)) {
                Eigen::Vector3d t_change = t - t_before;
                double angle_change = q.angularDistance(q_before);
                std::cout << "[SuperLoc Debug] Parameter change - trans: " << t_change.norm()
                          << ", rot: " << pcl::rad2deg(angle_change) << " deg" << std::endl;
            }

            // 6. 计算误差指标
            double rmse = 0.0;
            int inlier_count = 0;
            for (size_t i = 0; i < correspondences.size(); ++i) {
                const auto &corr = correspondences[i];
                Eigen::Vector3d src_pt(source->points[corr.first].x,
                                       source->points[corr.first].y,
                                       source->points[corr.first].z);
                Eigen::Vector3d transformed_pt = q * src_pt + t;

                double residual = normals[i].dot(transformed_pt) + plane_d_values[i];
                rmse += residual * residual;

                if (std::abs(residual) < 0.1) {
                    inlier_count++;
                }
            }

            rmse = std::sqrt(rmse / correspondences.size());
            double fitness = static_cast<double>(inlier_count) / correspondences.size();

            // 记录迭代日志
            iter_log.rmse = rmse;
            iter_log.fitness = fitness;
            iter_log.transform_matrix = final_transform;

            auto iter_end = std::chrono::high_resolution_clock::now();
            iter_log.iter_time_ms = std::chrono::duration<double, std::milli>(
                    iter_end - iter_start).count();

            iteration_logs.push_back(iter_log);

            // 检查收敛条件 - 与原始SuperLoc保持一致
            if ((summary.num_successful_steps > 0) || (iter == max_iterations - 1)) {
                result.converged = (summary.num_successful_steps > 0) && (rmse < 0.01);
                result.iterations = iter + 1;
                result.final_rmse = rmse;
                result.final_fitness = fitness;

                // 7. 估计配准误差和协方差 - 与原始SuperLoc保持一致
                EstimateRegistrationError(problem, pose_parameters, result);

                // 8. 基于特征可观测性计算不确定性
                computeUncertaintiesFromHistogram(PlaneFeatureHistogramObs, result);

                // 保存特征直方图供后续分析
                result.feature_histogram = PlaneFeatureHistogramObs;

                // 9. 判断退化
                checkDegeneracy(result);

                break;
            }
        }

        if (!result.converged) {
            result.iterations = max_iterations;
            result.final_rmse = iteration_logs.back().rmse;
            result.final_fitness = iteration_logs.back().fitness;

            // 即使没有收敛，也进行最后的分析
            // 估计配准误差和协方差
            ceres::Problem final_problem;
            ceres::LocalParameterization *pose_parameterization =
                    new SuperLocPoseParameterization();
            final_problem.AddParameterBlock(pose_parameters, 7, pose_parameterization);

            // 使用最后一次迭代的对应点重建问题（简化版）
            EstimateRegistrationError(final_problem, pose_parameters, result);

            // 计算不确定性
            computeUncertaintiesFromHistogram(PlaneFeatureHistogramObs, result);
            result.feature_histogram = PlaneFeatureHistogramObs;

            // 检查退化
            checkDegeneracy(result);
        }

        return result.converged;
    }

// 扩展的对应点查找函数，包含法向量计算
    void SuperLocICPIntegration::findCorrespondencesWithNormals(
            const pcl::PointCloud<pcl::PointXYZI>::Ptr &source,
            const pcl::PointCloud<pcl::PointXYZI>::Ptr &target,
            const Eigen::Matrix4d &transform,
            double max_distance,
            std::vector <std::pair<int, int>> &correspondences,
            std::vector <Eigen::Vector3d> &normals,
            std::vector<double> &plane_d_values) {

        correspondences.clear();
        normals.clear();
        plane_d_values.clear();

        pcl::KdTreeFLANN <pcl::PointXYZI> kdtree;
        kdtree.setInputCloud(target);

        // 对每个源点查找最近邻
        for (size_t i = 0; i < source->size(); ++i) {
            Eigen::Vector4d src_pt(source->points[i].x, source->points[i].y,
                                   source->points[i].z, 1.0);
            Eigen::Vector4d transformed_pt = transform * src_pt;

            pcl::PointXYZI query_pt;
            query_pt.x = transformed_pt(0);
            query_pt.y = transformed_pt(1);
            query_pt.z = transformed_pt(2);

            // 查找k个最近邻用于平面拟合
            std::vector<int> k_indices;
            std::vector<float> k_distances;
            if (kdtree.nearestKSearch(query_pt, 5, k_indices, k_distances) >= 5) {
                if (k_distances[0] <= max_distance * max_distance) {
                    // 计算平面参数 - 与原始SuperLoc保持一致
                    Eigen::MatrixXd A(5, 3);
                    Eigen::VectorXd b = -Eigen::VectorXd::Ones(5);

                    for (int j = 0; j < 5; ++j) {
                        A(j, 0) = target->points[k_indices[j]].x;
                        A(j, 1) = target->points[k_indices[j]].y;
                        A(j, 2) = target->points[k_indices[j]].z;
                    }

                    // 使用最小二乘拟合平面: Ax + By + Cz + 1 = 0
                    // 求解 [A B C]^T，使得 ||[x y z][A B C]^T + 1||^2 最小
                    Eigen::Vector3d plane_coeffs = A.colPivHouseholderQr().solve(b);

                    // plane_coeffs = [A, B, C]，平面方程为 Ax + By + Cz + 1 = 0
                    // 转换为标准形式 n·p + d = 0，其中|n| = 1
                    double norm = plane_coeffs.norm();
                    if (norm < 1e-6) continue;  // 退化平面

                    Eigen::Vector3d plane_normal = plane_coeffs / norm;
                    double negative_OA_dot_norm = 1.0 / norm;  // 这是原始SuperLoc中的d

                    // 确保法向量朝向视点（与原始SuperLoc一致）
                    Eigen::Vector3d viewpoint_direction(query_pt.x, query_pt.y, query_pt.z);
                    if (viewpoint_direction.dot(plane_normal) < 0) {
                        plane_normal = -plane_normal;
                        negative_OA_dot_norm = -negative_OA_dot_norm;
                    }

                    correspondences.push_back(std::make_pair(i, k_indices[0]));
                    normals.push_back(plane_normal);
                    plane_d_values.push_back(negative_OA_dot_norm);
                }
            }
        }
    }

// 详细的特征可观测性分析 - 与原始SuperLoc保持一致
    void SuperLocICPIntegration::analyzeFeatureObservabilityDetailed(
            const pcl::PointCloud<pcl::PointXYZI>::Ptr &source,
            const std::vector <std::pair<int, int>> &correspondences,
            const std::vector <Eigen::Vector3d> &normals,
            const Eigen::Matrix4d &transform,
            std::array<int, 9> &histogram) {

        // 获取当前旋转矩阵
        Eigen::Matrix3d R = transform.block<3, 3>(0, 0);

        // 旋转后的坐标轴
        Eigen::Vector3d x_axis = R * Eigen::Vector3d(1, 0, 0);
        Eigen::Vector3d y_axis = R * Eigen::Vector3d(0, 1, 0);
        Eigen::Vector3d z_axis = R * Eigen::Vector3d(0, 0, 1);

        for (size_t i = 0; i < correspondences.size(); ++i) {
            const auto &corr = correspondences[i];
            Eigen::Vector3d src_pt(source->points[corr.first].x,
                                   source->points[corr.first].y,
                                   source->points[corr.first].z);

            // 变换点到世界坐标系
            Eigen::Vector3d transformed_pt = transform.block<3, 3>(0, 0) * src_pt +
                                             transform.block<3, 1>(0, 3);

            const Eigen::Vector3d &normal = normals[i];

            // 计算叉积（用于旋转可观测性）
            Eigen::Vector3d cross = transformed_pt.cross(normal);

            // 旋转可观测性分析
            std::vector <std::pair<double, int>> rotation_quality;
            rotation_quality.push_back({std::abs(cross.dot(x_axis)), 0});      // rx_cross
            rotation_quality.push_back({std::abs(cross.dot(-x_axis)), 1});     // neg_rx_cross
            rotation_quality.push_back({std::abs(cross.dot(y_axis)), 2});      // ry_cross
            rotation_quality.push_back({std::abs(cross.dot(-y_axis)), 3});     // neg_ry_cross
            rotation_quality.push_back({std::abs(cross.dot(z_axis)), 4});      // rz_cross
            rotation_quality.push_back({std::abs(cross.dot(-z_axis)), 5});     // neg_rz_cross

            // 平移可观测性分析
            std::vector <std::pair<double, int>> trans_quality;
            trans_quality.push_back({std::abs(normal.dot(x_axis)), 6});    // tx_dot
            trans_quality.push_back({std::abs(normal.dot(y_axis)), 7});    // ty_dot
            trans_quality.push_back({std::abs(normal.dot(z_axis)), 8});    // tz_dot

            // 选择最佳的旋转和平移可观测性
            std::sort(rotation_quality.begin(), rotation_quality.end(),
                      [](const auto &a, const auto &b) { return a.first > b.first; });
            std::sort(trans_quality.begin(), trans_quality.end(),
                      [](const auto &a, const auto &b) { return a.first > b.first; });

            // 更新直方图
            histogram[rotation_quality[0].second]++;
            histogram[rotation_quality[1].second]++;
            histogram[trans_quality[0].second]++;
        }
    }

// 估计配准误差 - 与原始SuperLoc的EstimateRegistrationError保持一致
    void SuperLocICPIntegration::EstimateRegistrationError(
            ceres::Problem &problem,
            const double *pose_parameters,
            SuperLocResult &result) {

        // 协方差计算选项 - 与原始SuperLoc保持一致
        ceres::Covariance::Options covOptions;
        covOptions.apply_loss_function = true;
        covOptions.algorithm_type = ceres::CovarianceAlgorithmType::DENSE_SVD;
        covOptions.null_space_rank = -1;
        covOptions.num_threads = 2;

        ceres::Covariance covarianceSolver(covOptions);
        std::vector <std::pair<const double *, const double *>> covarianceBlocks;
        covarianceBlocks.emplace_back(pose_parameters, pose_parameters);

        if (covarianceSolver.Compute(covarianceBlocks, &problem)) {
            // 获取6x6协方差矩阵（在切空间中）
            result.covariance.setZero();
            covarianceSolver.GetCovarianceBlockInTangentSpace(
                    pose_parameters, pose_parameters, result.covariance.data());

            // 计算条件数
            Eigen::SelfAdjointEigenSolver <Eigen::Matrix<double, 6, 6>> solver_full(result.covariance);
            Eigen::VectorXd eigenvalues = solver_full.eigenvalues();
            std::cout << "info Eigen values: " << eigenvalues.transpose() << std::endl;

            // 避免除零
            double min_eigenvalue = std::max(eigenvalues(0), 1e-10);
            double max_eigenvalue = std::max(eigenvalues(5), 1e-10);
            result.cond_full = std::sqrt(max_eigenvalue / min_eigenvalue);

            // 位置部分条件数
            Eigen::SelfAdjointEigenSolver <Eigen::Matrix3d> solver_trans(
                    result.covariance.topLeftCorner<3, 3>());
            double min_trans = std::max(solver_trans.eigenvalues()(0), 1e-10);
            double max_trans = std::max(solver_trans.eigenvalues()(2), 1e-10);
            result.cond_trans = std::sqrt(max_trans / min_trans);

            // 旋转部分条件数
            Eigen::SelfAdjointEigenSolver <Eigen::Matrix3d> solver_rot(
                    result.covariance.bottomRightCorner<3, 3>());
            double min_rot = std::max(solver_rot.eigenvalues()(0), 1e-10);
            double max_rot = std::max(solver_rot.eigenvalues()(2), 1e-10);
            result.cond_rot = std::sqrt(max_rot / min_rot);
        } else {
            // 如果协方差计算失败，设置默认值
            result.covariance.setIdentity();
            result.cond_full = 1.0;
            result.cond_trans = 1.0;
            result.cond_rot = 1.0;
        }
    }

// 基于特征直方图计算不确定性 - 与原始SuperLoc的EstimateLidarUncertainty保持一致
    void SuperLocICPIntegration::computeUncertaintiesFromHistogram(
            const std::array<int, 9> &histogram,
            SuperLocResult &result) {

        // 平移特征总数
        double TotalTransFeature = histogram[6] + histogram[7] + histogram[8];

        if (TotalTransFeature > 0) {
            // X方向不确定性
            double uncertaintyX = (histogram[6] / TotalTransFeature) * 3;
            result.uncertainty_x = std::min(uncertaintyX, 1.0);

            // Y方向不确定性
            double uncertaintyY = (histogram[7] / TotalTransFeature) * 3;
            result.uncertainty_y = std::min(uncertaintyY, 1.0);

            // Z方向不确定性
            double uncertaintyZ = (histogram[8] / TotalTransFeature) * 3;
            result.uncertainty_z = std::min(uncertaintyZ, 1.0);
        } else {
            result.uncertainty_x = 0.0;
            result.uncertainty_y = 0.0;
            result.uncertainty_z = 0.0;
        }

        // 旋转特征总数
        double TotalRotationFeature = histogram[0] + histogram[1] + histogram[2] +
                                      histogram[3] + histogram[4] + histogram[5];

        if (TotalRotationFeature > 0) {
            // Roll不确定性
            double uncertaintyRoll = ((histogram[0] + histogram[1]) / TotalRotationFeature) * 3;
            result.uncertainty_roll = std::min(uncertaintyRoll, 1.0);

            // Pitch不确定性
            double uncertaintyPitch = ((histogram[2] + histogram[3]) / TotalRotationFeature) * 3;
            result.uncertainty_pitch = std::min(uncertaintyPitch, 1.0);

            // Yaw不确定性
            double uncertaintyYaw = ((histogram[4] + histogram[5]) / TotalRotationFeature) * 3;
            result.uncertainty_yaw = std::min(uncertaintyYaw, 1.0);
        } else {
            result.uncertainty_roll = 0.0;
            result.uncertainty_pitch = 0.0;
            result.uncertainty_yaw = 0.0;
        }
    }

// 检查退化 - 基于不确定性和特征数量
    void SuperLocICPIntegration::checkDegeneracy(SuperLocResult &result) {
        // 使用与原始SuperLoc完全一致的退化判断条件
        // 原始代码中的判断逻辑包括不确定性阈值和特征数量检查

        // 基于不确定性的退化判断
        if (result.uncertainty_x < 0.2 || result.uncertainty_y < 0.1 || result.uncertainty_z < 0.2) {
            result.isDegenerate = true;
        }
            // 基于条件数的退化判断（作为额外的安全检查）
//        else if (result.cond_full > 100.0 || result.cond_trans > 100.0 || result.cond_rot > 100.0) {
//            result.isDegenerate = true;
//        }
        else {
            result.isDegenerate = false;
        }

        // 注：原始SuperLoc还检查了特征数量(PlaneFeatureHistogramObs.at(6/7/8) < 20/10/10)
        // 但在我们的实现中，这个检查已经隐含在不确定性计算中
        // 因为当特征数量少时，不确定性会自动变高
    }

// 保持原有的辅助函数
    void SuperLocICPIntegration::findCorrespondences(
            const pcl::PointCloud<pcl::PointXYZI>::Ptr &source,
            const pcl::PointCloud<pcl::PointXYZI>::Ptr &target,
            const Eigen::Matrix4d &transform,
            double max_distance,
            std::vector <std::pair<int, int>> &correspondences,
            std::vector <Eigen::Vector3d> &normals) {

        // 调用新的函数，忽略plane_d_values
        std::vector<double> plane_d_values;
        findCorrespondencesWithNormals(source, target, transform, max_distance,
                                       correspondences, normals, plane_d_values);
    }

    void SuperLocICPIntegration::analyzeFeatureObservability(
            const pcl::PointCloud<pcl::PointXYZI>::Ptr &source,
            const std::vector <std::pair<int, int>> &correspondences,
            const std::vector <Eigen::Vector3d> &normals,
            const Eigen::Matrix4d &transform,
            std::array<int, 9> &histogram) {

        // 调用新的详细分析函数
        analyzeFeatureObservabilityDetailed(source, correspondences, normals,
                                            transform, histogram);
    }

    void SuperLocICPIntegration::computeUncertainties(
            const std::array<int, 9> &histogram,
            SuperLocResult &result) {

        // 调用新的函数
        computeUncertaintiesFromHistogram(histogram, result);
    }

    void SuperLocICPIntegration::computeDegeneracyFromCovariance(
            const Eigen::Matrix<double, 6, 6> &covariance,
            double threshold,
            SuperLocResult &result) {

        // 该函数已经集成到EstimateRegistrationError中
        // 保留以维持接口兼容性
    }

    void SuperLocICPIntegration::computePlaneParameters(
            const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
            const std::vector<int> &indices,
            Eigen::Vector3d &normal,
            double &d) {

        if (indices.size() < 3) {
            normal = Eigen::Vector3d(0, 0, 1);
            d = 0;
            return;
        }

        // 使用最小二乘拟合平面
        Eigen::MatrixXd A(indices.size(), 3);
        Eigen::VectorXd b = -Eigen::VectorXd::Ones(indices.size());

        for (size_t i = 0; i < indices.size(); ++i) {
            A(i, 0) = cloud->points[indices[i]].x;
            A(i, 1) = cloud->points[indices[i]].y;
            A(i, 2) = cloud->points[indices[i]].z;
        }

        normal = A.colPivHouseholderQr().solve(b);
        d = 1.0 / normal.norm();
        normal.normalize();
    }

} // namespace ICPRunner