import { useState } from 'react';
import { Button } from '../components/UI.jsx';
import PageLayout from '../components/PageLayout.jsx';

const CHARACTERS = [
  { id: '1', name: 'Michelle', modelUrl: '/models/1.glb', imageUrl: '/images/char1.png' },
  { id: '2', name: 'Sophie', modelUrl: '/models/2.glb', imageUrl: '/images/char2.png' },
  { id: '3', name: 'Remy', modelUrl: '/models/3.glb', imageUrl: '/images/char3.png' },
];

export default function CharacterSelect({
  refChar,
  setRefChar,
  userChar,
  setUserChar,
  sameChar,
  setSameChar,
  navigate,
}) {
  const [step, setStep] = useState('ref');
  const currentSelected = step === 'ref' ? refChar : userChar;

  const handleSelect = (id) => {
    if (step === 'ref') {
      setRefChar(id);
      if (sameChar) {
        setUserChar(id);
      }
    } else {
      setUserChar(id);
    }
  };

  const handleToggleSame = (checked) => {
    setSameChar(checked);
    if (checked) {
      setUserChar(refChar);
    }
  };

  const handleNext = () => {
    if (step === 'ref') {
      if (sameChar) {
        navigate('ready');
      } else {
        setStep('user');
        window.scrollTo(0, 0);
      }
    } else {
      navigate('ready');
    }
  };

  const handleBack = () => {
    if (step === 'user') {
      setStep('ref');
      window.scrollTo(0, 0);
    } else {
      navigate('settings');
    }
  };

  return (
    <PageLayout
      title="3D 캐릭터 선택"
      onBack={handleBack}
      scrollLabel="3D 캐릭터 선택"
      actions={
        <Button onClick={handleNext}>
          다음 <span>→</span>
        </Button>
      }
    >
      <h2 className="section-title" style={{ marginTop: 0 }}>
        {step === 'ref' ? '1. 기준 안무 캐릭터 선택' : '2. 내 캐릭터 선택 (웹캠 리타겟팅)'}
      </h2>

      {/* 3개 나란히 카드 그리드: 통일된 960px 페이지 높이에 맞춰 한 화면에 들어오도록 비율 조정 */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
          gap: '16px',
          marginTop: '8px',
        }}
      >
        {CHARACTERS.map((char) => {
          const isSelected = currentSelected === char.id;
          return (
            <button
              key={char.id}
              type="button"
              aria-pressed={isSelected}
              onClick={() => handleSelect(char.id)}
              style={{
                position: 'relative',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                padding: '12px',
                borderRadius: '16px',
                border: isSelected ? '2px solid #6c2bd9' : '1px solid #e2dce8',
                backgroundColor: isSelected ? '#faf7ff' : '#ffffff',
                boxShadow: isSelected ? '0 6px 18px rgba(108, 43, 217, 0.12)' : '0 2px 8px rgba(0, 0, 0, 0.03)',
                cursor: 'pointer',
                transition: 'all 0.18s ease',
              }}
            >
              {isSelected && (
                <span
                  style={{
                    position: 'absolute',
                    top: '10px',
                    right: '10px',
                    width: '24px',
                    height: '24px',
                    borderRadius: '50%',
                    backgroundColor: '#6c2bd9',
                    color: '#ffffff',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '13px',
                    fontWeight: 700,
                    zIndex: 2,
                  }}
                >
                  ✓
                </span>
              )}

              {/* 한 화면에 여유 있게 들어오는 4:4.5(정사각형에 가까운 세로형) 비율 */}
              <div
                style={{
                  width: '100%',
                  height: 'clamp(180px, calc(100dvh - 430px), 280px)',
                  borderRadius: '12px',
                  overflow: 'hidden',
                  backgroundColor: '#f3f0f8',
                  marginBottom: '10px',
                }}
              >
                <img
                  src={char.imageUrl}
                  alt={char.name}
                  style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'cover',
                    objectPosition: 'top center',
                    display: 'block',
                  }}
                />
              </div>

              <strong
                style={{
                  fontSize: '16px',
                  fontWeight: 700,
                  color: isSelected ? '#6c2bd9' : '#1f1b24',
                }}
              >
                {char.name}
              </strong>
            </button>
          );
        })}
      </div>

      {step === 'ref' && (
        <div
          style={{
            marginTop: '14px',
            padding: '12px 16px',
            borderRadius: '12px',
            backgroundColor: '#f7f5fb',
          }}
        >
          <label
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: 600,
              color: '#2d2638',
            }}
          >
            <input
              type="checkbox"
              checked={sameChar}
              onChange={(e) => handleToggleSame(e.target.checked)}
              style={{
                width: '17px',
                height: '17px',
                accentColor: '#6c2bd9',
                cursor: 'pointer',
              }}
            />
            <span>내 캐릭터도 기준 안무와 동일하게 설정</span>
          </label>
        </div>
      )}
    </PageLayout>
  );
}