# Copyright (c) 2025 Dooseop Choi. All rights reserved.
#
# This source code is licensed under the GPL License found in the
# LICENSE file in the root directory of this source tree.
# For more information, contact d1024.choi@etri.re.kr

from libraries import *
import torch
import sys
sys.path.append('../QCNetonETD')
from predictors.qcnet import QCNet

def load_trained_model(checkpoint_path):
    """학습된 QCNet 모델 로드"""
    model = QCNet.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model

def real_prediction_model(data, model, device, best_k, pred_len, obs_len):
    """실제 QCNet으로 예측 수행 - test_step과 동일한 방식"""
    model.eval()
    with torch.no_grad():
        # 데이터를 GPU로 이동 (HeteroData 객체인 경우)
        if torch.cuda.is_available() and hasattr(data, 'to'):
            data = data.to(device)
        elif torch.cuda.is_available() and isinstance(data, dict):
            # 딕셔너리 형태의 데이터를 device로 이동
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    data[key] = value.to(device)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, torch.Tensor):
                            data[key][sub_key] = sub_value.to(device)
        
        # 모델 예측
        pred = model(data)
        
        # test_step과 동일한 방식으로 traj_refine 구성
        if model.output_head:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :model.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :model.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :model.output_dim],
                                     pred['scale_refine_pos'][..., :model.output_dim]], dim=-1)
        
        # reorder modes by probability so that mode-0 is the most probable (test_step과 동일)
        if 'pi' in pred:
            with torch.no_grad():
                pi = torch.nn.functional.softmax(pred['pi'], dim=-1)  # [N, M]
                order = torch.argsort(pi, dim=-1, descending=True)  # [N, M]
                # gather along mode dimension (dim=1)
                gather_index = order.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, traj_refine.size(2), traj_refine.size(3))
                traj_refine = torch.gather(traj_refine, dim=1, index=gather_index)
        
        # agent-centric to global coordinate system (test_step과 완전 동일)
        num_historical_steps = model.num_historical_steps
        origin_eval = data['agent']['position'][:, num_historical_steps - 1]
        theta_eval = data['agent']['heading'][:, num_historical_steps - 1]
        cos, sin = theta_eval.cos(), theta_eval.sin()
        rot_mat = torch.zeros(data['agent']['num_nodes'], 2, 2, device=device)
        rot_mat[:, 0, 0], rot_mat[:, 0, 1] = cos, sin
        rot_mat[:, 1, 0], rot_mat[:, 1, 1] = -sin, cos
        traj_eval_pred = torch.matmul(traj_refine[:, :, :, :2], rot_mat.unsqueeze(1)) \
                         + origin_eval[:, :2].reshape(-1, 1, 1, 2)
        traj_eval_pred = traj_eval_pred.cpu()
        
        # numpy로 변환
        predictions = traj_eval_pred.numpy()
        
    return predictions

def main():

    # parameter setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, default='/workspace/ETRITrajPredChallenage/competition_data/test_qcnet', help='DO NOT CHANGE THIS!!')
    parser.add_argument('--save_path', type=str, default='/workspace/ETRITrajPredChallenage/prediction_results/', help='Directory to save prediction results')
    parser.add_argument('--checkpoint_path', type=str, default='/workspace/ETRITrajPredChallenage/yong.ckpt', help='Path to the checkpoint file, it should be based on docker container pathing not local pathing')
    parser.add_argument('--past_horizon_seconds', type=float, default=2, help='DO NOT CHANGE THIS!!')
    parser.add_argument('--future_horizon_seconds', type=float, default=6, help='DO NOT CHANGE THIS!!')
    parser.add_argument('--target_sample_period', type=float, default=10, help='DO NOT CHANGE THIS!!')
    parser.add_argument('--best_k', type=int, default=6, help='DO NOT CHANGE THIS!!')
    args = parser.parse_args()

    obs_len = args.past_horizon_seconds * args.target_sample_period
    pred_len = args.future_horizon_seconds * args.target_sample_period

    # 체크포인트 파일명에서 폴더명 생성
    ckpt_filename = os.path.basename(args.checkpoint_path)
    ckpt_name = os.path.splitext(ckpt_filename)[0]
    
    # 저장 디렉토리를 체크포인트 이름으로 생성
    final_save_path = os.path.join(args.save_path, ckpt_name)
    os.makedirs(final_save_path, exist_ok=True)
    
    print(f"Predictions will be saved to: {final_save_path}")
    
    # 학습된 모델 로드
    print(f"Loading trained model from: {args.checkpoint_path}")
    model = load_trained_model(args.checkpoint_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # transform each raw tracking file to driving scenes to Argoverse2 driving scenes
    file_names = [f for f in os.listdir(args.source_path) if f.endswith('.pkl')]
    for idx, file_name in enumerate(tqdm(file_names, desc="Generating predictions")):

        # read gt data
        with open(os.path.join(args.source_path, file_name), 'rb') as f:
            data = pickle.load(f)

        # do prediction with trained QCNet model
        predictions = real_prediction_model(data, model, device, args.best_k, pred_len, obs_len)

        # test_step과 동일한 방식으로 scene별로 저장 
        start_end_indices = data['agent']['ptr'] if 'ptr' in data['agent'] else [0, data['agent']['num_nodes']]
        
        if isinstance(start_end_indices, torch.Tensor):
            start_end_indices = start_end_indices.cpu().numpy()
        elif not isinstance(start_end_indices, (list, np.ndarray)):
            # 단일 scene인 경우
            start_end_indices = [0, data['agent']['num_nodes']]

        for scene_idx, (start, end) in enumerate(zip(start_end_indices[:-1], start_end_indices[1:])):
            agent = {
                'num_nodes': end - start,
                'num_valid_nodes': data['agent']['num_valid_nodes'][scene_idx] if isinstance(data['agent']['num_valid_nodes'], (list, np.ndarray)) else data['agent']['num_valid_nodes'],
                'id': data['agent']['id'][scene_idx] if isinstance(data['agent']['id'], (list, np.ndarray)) else data['agent']['id'],
                'category': data['agent']['category'][scene_idx] if isinstance(data['agent']['category'], (list, np.ndarray)) else data['agent']['category'],
                'predictions': predictions[start:end]
            }

            scene = {
                'log_id': data['log_id'][scene_idx] if isinstance(data['log_id'], (list, np.ndarray)) else data['log_id'],
                'frm_idx': data['frm_idx'][scene_idx] if isinstance(data['frm_idx'], (list, np.ndarray, torch.Tensor)) else data['frm_idx'],
                'agent': agent
            }

            # save data
            file_name_submission = file_name.replace('_masked_qcnet.pkl', '_submission.pkl')
            with open(os.path.join(final_save_path, file_name_submission), 'wb') as f:
                pickle.dump(scene, f)


if __name__ == '__main__':
    main()

